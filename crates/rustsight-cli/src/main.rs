use clap::{Parser, Subcommand};
use rustsight_core::types::*;
use rustsight_core::{Engine, SystemClock};
use rustsight_providers::{
    build_mock_providers, build_operation_llms, build_providers, retain_max_completion_tokens,
    OperationLlms, ProviderEnv, SimpleTokenEstimator,
};
use rustsight_server::{auth::ApiAuth, http, mcp};
use rustsight_storage::{SqliteStore, StoreOptions};
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
        #[arg(long, required_unless_present = "file", conflicts_with = "file")]
        content: Option<String>,
        /// Retain a file instead: parsed like the files/retain API (text,
        /// html), its bytes kept as an attachment of the document.
        #[arg(long)]
        file: Option<std::path::PathBuf>,
        /// Parser chain for --file, comma-separated (default
        /// RUSTSIGHT_FILE_PARSER, else text,html).
        #[arg(long, requires = "file")]
        parser: Option<String>,
        #[arg(long)]
        document_id: Option<String>,
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
    /// Bank templates: export a bank's settings, mental models and
    /// directives, or import a manifest into a bank.
    Template {
        #[command(subcommand)]
        action: TemplateCmd,
    },
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
        /// Memory Defense policy as JSON, e.g.
        /// '{"enabled":true,"rules":[{"on":"sensitive_data","action":"redact"}]}';
        /// `off` removes it.
        #[arg(long)]
        memory_defense: Option<String>,
    },
    /// Rebuild the sqlite-vec KNN index from stored embeddings (RS-7): the
    /// migration path for banks written before the index existed or by a
    /// build without it. Omit --bank to reindex every bank.
    Reindex {
        #[arg(long)]
        bank: Option<String>,
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

#[derive(Subcommand)]
enum TemplateCmd {
    /// Print the bank's template manifest.
    Export {
        #[arg(long)]
        bank: String,
    },
    /// Apply a manifest file to the bank (created if missing).
    Import {
        #[arg(long)]
        bank: String,
        #[arg(long)]
        file: PathBuf,
        /// Validate and report without changing anything.
        #[arg(long)]
        dry_run: bool,
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

/// `RUSTSIGHT_VECTOR_INDEX` (default on): `off`/`false`/`0`/`no` keeps
/// semantic search on the brute-force scan.
fn store_options() -> StoreOptions {
    let vector_index = match std::env::var("RUSTSIGHT_VECTOR_INDEX") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "" | "on" | "true" | "1" | "yes" => true,
            "off" | "false" | "0" | "no" => false,
            other => {
                eprintln!("rustsight: invalid RUSTSIGHT_VECTOR_INDEX={other:?}, using default on");
                true
            }
        },
        Err(_) => true,
    };
    StoreOptions { vector_index }
}

fn open_store(db: &PathBuf, clock: Arc<SystemClock>) -> SqliteStore {
    SqliteStore::open_with_options(db, clock, &store_options()).unwrap_or_else(|e| {
        eprintln!("failed to open database {}: {e}", db.display());
        std::process::exit(1);
    })
}

async fn build_engine(db: &PathBuf, mock: bool) -> Arc<Engine> {
    let clock = Arc::new(SystemClock);
    let store = open_store(db, clock.clone());
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
    match rustsight_server::default_bank_template_from_env() {
        Ok(Some(template)) => engine = engine.with_default_template(template),
        Ok(None) => {}
        Err(msg) => {
            eprintln!("configuration error: {msg}");
            std::process::exit(1);
        }
    }
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
    // Store maintenance needs no providers (and no API key).
    if let Cmd::Reindex { bank } = &cli.cmd {
        let store = open_store(&db, Arc::new(SystemClock));
        match store.reindex_vectors(bank.as_deref()) {
            Ok(reports) => {
                for r in reports.iter().filter(|r| r.skipped > 0) {
                    eprintln!(
                        "rustsight: bank {} has {} embedding(s) wider than vec0 supports; it stays on brute force",
                        r.bank_id, r.skipped
                    );
                }
                print_json(&reports);
                return;
            }
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
    }
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
            file,
            parser,
            document_id,
            context,
            timestamp_ms,
        } => {
            let req = RetainRequest {
                bank_id: bank,
                content: content.unwrap_or_default(),
                context,
                timestamp_ms,
                document_id,
                ..Default::default()
            };
            match file {
                Some(path) => retain_file(&engine, req, &path, parser.as_deref()).await,
                None => engine.retain(req).await,
            }
            .map(|r| print_json(&r))
        }
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
        Cmd::Template { action } => match action {
            TemplateCmd::Export { bank } => match engine.export_template(&bank) {
                Ok(Some(manifest)) => {
                    print_json(&manifest);
                    Ok(())
                }
                Ok(None) => Err(rustsight_core::Error::NotFound(format!("bank {bank}"))),
                Err(e) => Err(e),
            },
            TemplateCmd::Import {
                bank,
                file,
                dry_run,
            } => std::fs::read_to_string(&file)
                .map_err(|e| {
                    rustsight_core::Error::BadInput(format!("reading {}: {e}", file.display()))
                })
                .and_then(|text| {
                    serde_json::from_str(&text)
                        .map_err(|e| rustsight_core::Error::BadInput(format!("invalid JSON: {e}")))
                })
                .and_then(|raw| rustsight_core::templates::parse_manifest(&raw))
                .and_then(|manifest| engine.import_template(&bank, &manifest, dry_run))
                .map(|report| print_json(&report)),
        },
        Cmd::Reindex { .. } => unreachable!("handled before the engine is built"),
        Cmd::BankSet {
            bank,
            mission,
            directives,
            skepticism,
            literalism,
            empathy,
            memory_defense,
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
            if let Some(raw) = memory_defense {
                let mut config = match engine.store.bank_config(&bank)? {
                    serde_json::Value::Object(map) => map,
                    _ => serde_json::Map::new(),
                };
                if raw == "off" {
                    config.remove("memory_defense");
                } else {
                    let policy: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
                        rustsight_core::Error::BadInput(format!("--memory-defense: {e}"))
                    })?;
                    rustsight_core::defense::parse_policy(Some(&policy))?;
                    config.insert("memory_defense".into(), policy);
                }
                engine
                    .store
                    .set_bank_config(&bank, &serde_json::Value::Object(config))?;
            }
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

/// Media type for an attachment, from the file extension.
fn guess_content_type(path: &std::path::Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "txt" | "log" => "text/plain",
        "md" | "markdown" => "text/markdown",
        "csv" => "text/csv",
        "tsv" => "text/tab-separated-values",
        "json" => "application/json",
        "yaml" | "yml" => "application/yaml",
        "xml" => "application/xml",
        "html" | "htm" => "text/html",
        "pdf" => "application/pdf",
        _ => return None,
    })
}

/// `retain --file`: the files/retain flow, run in-process — store the
/// bytes as an attachment, convert with the parser chain, retain the text
/// with `attachment_id` / `filename` metadata.
async fn retain_file(
    engine: &Engine,
    mut req: RetainRequest,
    path: &std::path::Path,
    parser: Option<&str>,
) -> rustsight_core::Result<rustsight_core::RetainResponse> {
    use rustsight_core::files::{Attachment, FileInput};
    let bytes = std::fs::read(path)
        .map_err(|e| rustsight_core::Error::BadInput(format!("reading {}: {e}", path.display())))?;
    let filename = path
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into());
    let chain: Vec<String> = match parser {
        Some(raw) => raw
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect(),
        None => rustsight_server::files::FileSettings::from_env().default_chain,
    };
    let content_type = guess_content_type(path);
    let converted = engine.parsers.convert_with_fallback(
        &chain,
        &FileInput {
            bytes: &bytes,
            filename: &filename,
            content_type,
        },
    )?;
    let document_id = req
        .document_id
        .clone()
        .unwrap_or_else(|| format!("file_{}", uuid::Uuid::new_v4()));
    let attachment = Attachment::new(
        &req.bank_id,
        &document_id,
        &filename,
        content_type,
        bytes,
        engine.clock.now_ms(),
    );
    engine.store.put_attachment(&attachment)?;
    req.content = converted.text;
    req.document_id = Some(document_id.clone());
    let metadata = req.metadata.get_or_insert_with(Default::default);
    metadata.insert("attachment_id".into(), attachment.id.clone());
    metadata.insert("filename".into(), filename);
    let resp = engine.retain(req).await?;
    engine
        .store
        .prune_document_attachments(&attachment.bank_id, &document_id, &attachment.id)?;
    Ok(resp)
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

    #[test]
    fn file_content_types_follow_the_extension() {
        let ct = |p: &str| super::guess_content_type(std::path::Path::new(p));
        assert_eq!(ct("a/Notes.MD"), Some("text/markdown"));
        assert_eq!(ct("page.htm"), Some("text/html"));
        assert_eq!(ct("blob"), None);
    }
}
