use clap::{Parser, Subcommand};
use rustsight_core::types::*;
use rustsight_core::{Engine, SystemClock};
use rustsight_providers::{
    build_mock_providers, build_operation_llms, build_providers, retain_max_completion_tokens,
    OperationLlms, ProviderEnv, SimpleTokenEstimator,
};
use rustsight_server::{auth::ApiAuth, http, mcp};
use rustsight_storage::SqliteStore;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "rustsight",
    version,
    about = "Hindsight-style agent memory engine"
)]
struct Cli {
    #[arg(long, global = true)]
    db: Option<PathBuf>,

    #[arg(long, global = true)]
    mock: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Serve {
        #[arg(long, default_value = "127.0.0.1:8888")]
        addr: String,
        /// Bind host; with `--port`, the `hindsight-api`-style spelling of
        /// `--addr` (either flag overrides that part of `--addr`).
        #[arg(long)]
        host: Option<String>,
        #[arg(long)]
        port: Option<u16>,
        /// Accepted for `hindsight-api` command-line compatibility; rustsight
        /// logs to stderr at a single level.
        #[arg(long)]
        log_level: Option<String>,
        /// Serve on non-loopback interfaces without an API token
        /// (RUSTSIGHT_API_TOKEN / RUSTSIGHT_API_TOKEN_FILE).
        #[arg(long)]
        allow_unauth: bool,
    },
    Mcp {
        #[arg(long)]
        bank: Option<String>,
    },
    Retain {
        #[arg(long)]
        bank: String,
        #[arg(long)]
        content: String,
        #[arg(long)]
        context: Option<String>,
        #[arg(long)]
        timestamp_ms: Option<i64>,
    },
    Recall {
        #[arg(long)]
        bank: String,
        #[arg(long)]
        query: String,
        #[arg(long, default_value_t = 1000)]
        budget: usize,
    },
    Reflect {
        #[arg(long)]
        bank: String,
        #[arg(long)]
        query: String,
    },
    Consolidate {
        #[arg(long)]
        bank: String,
        #[arg(long, default_value_t = 50)]
        max: usize,
    },
    Inspect {
        #[arg(long)]
        bank: String,
    },
    Banks,
    BankSet {
        #[arg(long)]
        bank: String,
        #[arg(long)]
        mission: Option<String>,
        #[arg(long)]
        directives: Option<String>,
        #[arg(long)]
        skepticism: Option<u8>,
        #[arg(long)]
        literalism: Option<u8>,
        #[arg(long)]
        empathy: Option<u8>,
    },
    MentalModel {
        #[arg(long)]
        bank: String,
        #[arg(long)]
        question: String,
        #[arg(long)]
        path: Option<String>,
    },
}

fn default_db() -> PathBuf {
    if let Ok(p) = std::env::var("RUSTSIGHT_DB") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    match std::env::var("HOME") {
        Ok(home) if !home.is_empty() => PathBuf::from(home).join(".rustsight/rustsight.db"),
        _ => PathBuf::from("rustsight.db"),
    }
}

async fn build_engine(db: &PathBuf, mock: bool) -> Arc<Engine> {
    let clock = Arc::new(SystemClock);
    let store = SqliteStore::open_with_clock(db, clock.clone()).unwrap_or_else(|e| {
        eprintln!("failed to open database {}: {e}", db.display());
        std::process::exit(1);
    });
    let mut operation_llms = OperationLlms {
        retain: None,
        reflect: None,
        consolidate: None,
    };
    let (llm, embed, rerank) = if mock {
        build_mock_providers()
    } else {
        let env = match ProviderEnv::from_env() {
            Ok(env) => env,
            Err(msg) => {
                eprintln!("configuration error: {msg}");
                std::process::exit(1);
            }
        };
        if env.llm_api_key.is_empty() {
            eprintln!(
                "no API key configured: set RUSTSIGHT_LLM_API_KEY (or use --mock for an offline demo)"
            );
            std::process::exit(1);
        }
        operation_llms = build_operation_llms(&env).unwrap_or_else(|msg| {
            eprintln!("provider setup failed: {msg}");
            std::process::exit(1);
        });
        build_providers(&env).await.unwrap_or_else(|msg| {
            eprintln!("provider setup failed: {msg}");
            std::process::exit(1);
        })
    };
    let mut engine = Engine::new(
        Arc::new(store),
        llm,
        embed,
        rerank,
        Arc::new(SimpleTokenEstimator),
        clock,
    );
    // Per-operation models (e.g. a small extractor, a large answerer).
    engine.retain_llm = operation_llms.retain;
    engine.reflect_llm = operation_llms.reflect;
    engine.consolidate_llm = operation_llms.consolidate;
    engine.config.retain_max_completion_tokens = Some(retain_max_completion_tokens());
    Arc::new(engine)
}

/// `--addr` with its host and/or port replaced by `--host` / `--port`.
fn serve_addr(addr: &str, host: Option<&str>, port: Option<u16>) -> String {
    if host.is_none() && port.is_none() {
        return addr.to_string();
    }
    let (addr_host, addr_port) = addr.rsplit_once(':').unwrap_or((addr, "8888"));
    let host = host.unwrap_or(addr_host);
    // A bare IPv6 literal needs brackets to carry a port.
    let host = if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host.to_string()
    };
    match port {
        Some(p) => format!("{host}:{p}"),
        None => format!("{host}:{addr_port}"),
    }
}

fn print_json<T: serde::Serialize>(value: &T) {
    println!(
        "{}",
        serde_json::to_string_pretty(value).unwrap_or_default()
    );
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let db = cli.db.clone().unwrap_or_else(default_db);
    let engine = build_engine(&db, cli.mock).await;
    let result: Result<(), rustsight_core::Error> = match cli.cmd {
        Cmd::Serve {
            addr,
            host,
            port,
            log_level: _,
            allow_unauth,
        } => {
            let addr = serve_addr(&addr, host.as_deref(), port);
            let auth = ApiAuth::from_env();
            let auth_note = match &auth {
                ApiAuth::Token(_) => " (bearer-token auth on)",
                ApiAuth::None => " (no auth; loopback only)",
            };
            eprintln!(
                "rustsight listening on http://{addr} (db: {}){auth_note}",
                db.display()
            );
            http::serve(engine, &addr, auth, allow_unauth).await
        }
        Cmd::Mcp { bank } => mcp::run_stdio(engine, bank).await,
        Cmd::Retain {
            bank,
            content,
            context,
            timestamp_ms,
        } => engine
            .retain(RetainRequest {
                bank_id: bank,
                content,
                context,
                timestamp_ms,
                ..Default::default()
            })
            .await
            .map(|r| print_json(&r)),
        Cmd::Recall {
            bank,
            query,
            budget,
        } => engine
            .recall(RecallRequest {
                bank_id: bank,
                query,
                top_k_per_arm: engine.config.recall_budget.resolve(None, budget),
                token_budget: budget,
                kinds: None,
                ..Default::default()
            })
            .await
            .map(|r| print_json(&r)),
        Cmd::Reflect { bank, query } => engine.reflect(&bank, &query).await.map(|r| print_json(&r)),
        Cmd::Consolidate { bank, max } => engine
            .consolidate_once(&bank, max)
            .await
            .map(|n| println!("{{\"observations_updated\": {n}}}")),
        Cmd::Inspect { bank } => (|| -> rustsight_core::Result<()> {
            let memories = engine.store.list_memories(&bank, 500)?;
            let entities = engine.store.entities_for_bank(&bank)?;
            let observations = engine.store.observations_for_bank(&bank)?;
            let opinions = engine.store.list_opinions(&bank)?;
            print_json(&serde_json::json!({
                "bank": bank,
                "memories": memories,
                "entities": entities,
                "observations": observations,
                "opinions": opinions
            }));
            Ok(())
        })(),
        Cmd::Banks => engine.store.list_banks().map(|banks| print_json(&banks)),
        Cmd::BankSet {
            bank,
            mission,
            directives,
            skepticism,
            literalism,
            empathy,
        } => (|| -> rustsight_core::Result<()> {
            let existing = engine.store.get_bank(&bank)?.unwrap_or(Bank {
                id: bank.clone(),
                mission: None,
                directives: None,
                background: None,
                disposition: Disposition::default(),
            });
            let mut b = existing;
            if mission.is_some() {
                b.mission = mission;
            }
            if directives.is_some() {
                b.directives = directives;
            }
            if let Some(v) = skepticism {
                b.disposition.skepticism = v.clamp(1, 5);
            }
            if let Some(v) = literalism {
                b.disposition.literalism = v.clamp(1, 5);
            }
            if let Some(v) = empathy {
                b.disposition.empathy = v.clamp(1, 5);
            }
            engine.store.ensure_bank(&b)?;
            engine.store.update_bank(&b)?;
            print_json(&b);
            Ok(())
        })(),
        Cmd::MentalModel {
            bank,
            question,
            path,
        } => engine
            .refresh_mental_model(&bank, &question, path.as_deref())
            .await
            .map(|m| print_json(&m)),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::serve_addr;

    #[test]
    fn host_and_port_flags_override_parts_of_addr() {
        assert_eq!(serve_addr("127.0.0.1:8888", None, None), "127.0.0.1:8888");
        assert_eq!(
            serve_addr("127.0.0.1:8888", None, Some(9000)),
            "127.0.0.1:9000"
        );
        assert_eq!(
            serve_addr("127.0.0.1:8888", Some("0.0.0.0"), None),
            "0.0.0.0:8888"
        );
        assert_eq!(
            serve_addr("127.0.0.1:8888", Some("localhost"), Some(18790)),
            "localhost:18790"
        );
        assert_eq!(
            serve_addr("127.0.0.1:8888", Some("::1"), Some(1)),
            "[::1]:1"
        );
    }
}
