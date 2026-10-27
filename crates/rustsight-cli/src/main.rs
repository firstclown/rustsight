use clap::{Parser, Subcommand};
use rustsight_core::types::*;
use rustsight_core::{Engine, SystemClock};
use rustsight_providers::{build_mock_providers, build_providers, ProviderEnv, SimpleTokenEstimator};
use rustsight_storage::SqliteStore;
use rustsight_server::{http, mcp};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "rustsight", version, about = "Hindsight-style agent memory engine")]
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

fn build_engine(db: &PathBuf, mock: bool) -> Arc<Engine> {
    let store = SqliteStore::open(db).unwrap_or_else(|e| {
        eprintln!("failed to open database {}: {e}", db.display());
        std::process::exit(1);
    });
    let (llm, embed, rerank) = if mock {
        build_mock_providers()
    } else {
        let env = ProviderEnv::from_env();
        if env.llm_api_key.is_empty() {
            eprintln!(
                "no API key configured: set RUSTSIGHT_LLM_API_KEY (or use --mock for an offline demo)"
            );
            std::process::exit(1);
        }
        build_providers(&env)
    };
    Arc::new(Engine::new(
        Arc::new(store),
        llm,
        embed,
        rerank,
        Arc::new(SimpleTokenEstimator),
        Arc::new(SystemClock),
    ))
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
    let engine = build_engine(&db, cli.mock);
    let result: Result<(), rustsight_core::Error> = match cli.cmd {
        Cmd::Serve { addr } => {
            eprintln!("rustsight listening on http://{addr} (db: {})", db.display());
            http::serve(engine, &addr).await
        }
        Cmd::Mcp { bank } => mcp::run_stdio(engine, bank).await,
        Cmd::Retain { bank, content, context, timestamp_ms } => engine
            .retain(RetainRequest {
                bank_id: bank,
                content,
                context,
                timestamp_ms,
            })
            .await
            .map(|r| print_json(&r)),
        Cmd::Recall { bank, query, budget } => engine
            .recall(RecallRequest {
                bank_id: bank,
                query,
                token_budget: budget,
                top_k_per_arm: 10,
                kinds: None,
            })
            .await
            .map(|r| print_json(&r)),
        Cmd::Reflect { bank, query } => engine
            .reflect(&bank, &query)
            .await
            .map(|r| print_json(&r)),
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
        Cmd::Banks => engine
            .store
            .list_banks()
            .map(|banks| print_json(&banks)),
        Cmd::BankSet { bank, mission, directives, skepticism, literalism, empathy } => {
            (|| -> rustsight_core::Result<()> {
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
            })()
        }
        Cmd::MentalModel { bank, question, path } => engine
            .refresh_mental_model(&bank, &question, path.as_deref())
            .await
            .map(|m| print_json(&m)),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
