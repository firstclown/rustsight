//! Per-bank coordination between bank deletion and the work writing to it.
//!
//! Retain, consolidation, opinion formation and mental-model refreshes hold
//! a shared guard while they run; deleting a bank takes it exclusively, so
//! nothing can write a bank back into existence halfway through the delete.
//! Those jobs spend most of their time in provider calls, so a delete also
//! raises a flag that cancels them at their next await instead of waiting
//! minutes for an LLM answer nobody will use. Store writes happen between
//! awaits, so cancellation never splits one.

use crate::{Error, Result};
use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use tokio::sync::{watch, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

struct BankGate {
    lock: Arc<RwLock<()>>,
    deleting: watch::Sender<bool>,
}

#[derive(Default)]
pub(crate) struct BankGates {
    gates: Mutex<HashMap<String, Arc<BankGate>>>,
}

/// Shared access to a bank. Work run through [`BankGuard::until_deleted`]
/// stops if the bank is deleted.
pub struct BankGuard {
    _read: OwnedRwLockReadGuard<()>,
    deleting: watch::Receiver<bool>,
    bank_id: String,
}

impl BankGuard {
    pub fn is_deleting(&self) -> bool {
        *self.deleting.borrow()
    }

    pub async fn until_deleted<T>(&self, work: impl Future<Output = Result<T>>) -> Result<T> {
        let mut deleting = self.deleting.clone();
        tokio::select! {
            result = work => result,
            _ = deleting.wait_for(|flag| *flag) => {
                Err(Error::NotFound(format!("bank {} was deleted", self.bank_id)))
            }
        }
    }
}

/// Exclusive access to a bank. Dropping it lets queued work proceed.
pub(crate) struct ExclusiveBank {
    write: Option<OwnedRwLockWriteGuard<()>>,
    gate: Arc<BankGate>,
}

impl Drop for ExclusiveBank {
    fn drop(&mut self) {
        // The flag must fall before the lock opens, or the next reader
        // would be cancelled by a delete that has already finished.
        self.gate.deleting.send_replace(false);
        self.write.take();
    }
}

/// Clears the flag if the delete is abandoned while waiting for the lock
/// (a disconnected client); otherwise every later retain would be cancelled.
struct ClearOnAbandon<'a>(&'a BankGate, bool);

impl Drop for ClearOnAbandon<'_> {
    fn drop(&mut self) {
        if self.1 {
            self.0.deleting.send_replace(false);
        }
    }
}

impl BankGates {
    fn gate(&self, bank_id: &str) -> Arc<BankGate> {
        let mut gates = self.gates.lock().unwrap_or_else(|e| e.into_inner());
        Arc::clone(gates.entry(bank_id.to_string()).or_insert_with(|| {
            Arc::new(BankGate {
                lock: Arc::new(RwLock::new(())),
                deleting: watch::Sender::new(false),
            })
        }))
    }

    pub(crate) async fn read(&self, bank_id: &str) -> BankGuard {
        let gate = self.gate(bank_id);
        let read = Arc::clone(&gate.lock).read_owned().await;
        BankGuard {
            _read: read,
            deleting: gate.deleting.subscribe(),
            bank_id: bank_id.to_string(),
        }
    }

    pub(crate) async fn exclusive(&self, bank_id: &str) -> ExclusiveBank {
        let gate = self.gate(bank_id);
        gate.deleting.send_replace(true);
        let mut abandon = ClearOnAbandon(&gate, true);
        let write = Arc::clone(&gate.lock).write_owned().await;
        abandon.1 = false;
        drop(abandon);
        ExclusiveBank {
            write: Some(write),
            gate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn delete_cancels_running_work_and_later_work_proceeds() {
        let gates = Arc::new(BankGates::default());
        let guard = gates.read("b").await;
        let running = tokio::spawn(async move {
            guard
                .until_deleted(std::future::pending::<Result<()>>())
                .await
        });
        tokio::task::yield_now().await;
        let deletion = tokio::time::timeout(Duration::from_secs(5), gates.exclusive("b"))
            .await
            .expect("delete must not wait for cancelled work");
        assert!(matches!(running.await.unwrap(), Err(Error::NotFound(_))));
        drop(deletion);
        let next = gates.read("b").await;
        assert!(!next.is_deleting());
        assert_eq!(next.until_deleted(async { Ok(7) }).await.unwrap(), 7);
    }

    #[tokio::test]
    async fn an_abandoned_delete_does_not_poison_the_bank() {
        let gates = BankGates::default();
        let guard = gates.read("b").await;
        assert!(
            tokio::time::timeout(Duration::from_millis(10), gates.exclusive("b"))
                .await
                .is_err()
        );
        assert!(!guard.is_deleting());
    }
}
