use clap::{Parser, Subcommand};
use rustsight_eval::compare as record_compare;
use rustsight_eval::runner::{self, RunConfig};
use rustsight_eval::{analyze, dataset, download};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "rustsight-eval",
    version,
    about = "LongMemEval-S harness for the rustsight engine (RS-6)"
)]
struct Cli {
    /// Dataset variant; only the original S file is supported.
    #[arg(long, default_value = dataset::DEFAULT_VARIANT)]
    dataset: String,

    /// Output/run directory (state, banks.db, traces/, report.*).
    #[arg(long, default_value = "results")]
    out: PathBuf,

    /// Path to the dataset JSON (default ~/.rustsight/eval/longmemeval_s.json).
    #[arg(long)]
    dataset_path: Option<PathBuf>,

    /// Conversations processed in parallel during retain/answer.
    #[arg(long, default_value_t = 2)]
    conversations: usize,

    /// Only run the first N questions (smoke/staged runs).
    #[arg(long)]
    limit_questions: Option<usize>,

    /// Synthetic spacing between sessions, in days.
    #[arg(long, default_value_t = 1)]
    session_spacing_days: i64,

    /// Max dirty entities per consolidation pass.
    #[arg(long, default_value_t = 50)]
    consolidate_max: usize,

    /// Dirty entities synthesized concurrently inside one pass.
    #[arg(long, default_value_t = 4)]
    consolidate_concurrency: usize,

    /// Seconds between progress heartbeats on stderr (0 disables).
    #[arg(long, default_value_t = 60)]
    heartbeat_secs: u64,

    /// Offline end-to-end run with mock providers and a fixed judge.
    #[arg(long)]
    mock: bool,

    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the harness (default when no subcommand is given).
    Run,
    /// Download the dataset from HuggingFace.
    Download {
        /// Destination (default ~/.rustsight/eval/longmemeval_s.json).
        #[arg(long)]
        dest: Option<PathBuf>,
    },
    /// Summarize failure clusters from a run directory.
    Analyze {
        /// Run directory containing traces/.
        #[arg(long = "run")]
        run_dir: PathBuf,
    },
    /// Compare a run's per-question outcomes against the committed Hindsight
    /// LongMemEval-S record (parity check).
    Compare {
        /// Run directory containing traces/.
        #[arg(long = "run")]
        run_dir: PathBuf,
        /// Hindsight record JSON (default: the committed fixture).
        #[arg(long)]
        record: Option<PathBuf>,
    },
}

fn default_dataset_path() -> PathBuf {
    std::env::var("HOME")
        .ok()
        .filter(|h| !h.is_empty())
        .map(|h| PathBuf::from(h).join(".rustsight/eval/longmemeval_s.json"))
        .unwrap_or_else(|| PathBuf::from("longmemeval_s.json"))
}

/// `compare` subcommand: build the parity comparison against the Hindsight
/// record, print it, and persist it as `compare.md`/`compare.json` in the
/// run directory.
fn record_compare_cli(run_dir: &std::path::Path, record: &std::path::Path) -> Result<(), String> {
    let comparison = record_compare::compare(run_dir, record)?;
    let md = comparison.render_markdown();
    let json = serde_json::to_string_pretty(&comparison).map_err(|e| e.to_string())?;
    std::fs::write(run_dir.join("compare.md"), &md)
        .and_then(|_| std::fs::write(run_dir.join("compare.json"), json))
        .map_err(|e| format!("cannot write comparison outputs: {e}"))?;
    println!("{md}");
    eprintln!(
        "rustsight-eval: wrote {} and {}",
        run_dir.join("compare.md").display(),
        run_dir.join("compare.json").display()
    );
    Ok(())
}

#[tokio::main]
async fn main() {
    let mut cli = Cli::parse();
    let cmd = cli.cmd.take().unwrap_or(Cmd::Run);
    let result = match cmd {
        Cmd::Run => run_harness(cli).await,
        Cmd::Download { dest } => download::download(dest).await.map(|_| ()),
        Cmd::Analyze { run_dir } => analyze::analyze(&run_dir).map(|summary| {
            println!("{summary}");
        }),
        Cmd::Compare { run_dir, record } => {
            let record = record.unwrap_or_else(record_compare::default_record_path);
            record_compare_cli(&run_dir, &record)
        }
    };
    if let Err(e) = result {
        eprintln!("rustsight-eval: error: {e}");
        std::process::exit(1);
    }
}

async fn run_harness(cli: Cli) -> Result<(), String> {
    if cli.dataset != dataset::DEFAULT_VARIANT {
        return Err(format!(
            "unsupported dataset {:?}; only {} is available",
            cli.dataset,
            dataset::DEFAULT_VARIANT
        ));
    }
    let dataset_path = cli.dataset_path.unwrap_or_else(default_dataset_path);
    if !dataset_path.exists() {
        let dest = download::default_dest();
        return Err(format!(
            "dataset not found at {}. Run `cargo run -p rustsight-eval -- download` first \
             (fetches to {}), or pass --dataset-path",
            dataset_path.display(),
            dest.display()
        ));
    }
    let cfg = RunConfig {
        out: cli.out,
        dataset_path,
        conversations_parallel: cli.conversations,
        limit_questions: cli.limit_questions,
        session_spacing_days: cli.session_spacing_days,
        mock: cli.mock,
        consolidate_max: cli.consolidate_max,
        consolidate_concurrency: cli.consolidate_concurrency,
        heartbeat_secs: cli.heartbeat_secs,
        consolidate_pass_limit: 200,
    };
    runner::run(cfg).await.map(|_| ())
}
