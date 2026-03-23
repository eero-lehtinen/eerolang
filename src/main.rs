use crate::vm::Vm;
use bumpalo::Bump;
use clap::Parser;
use log::{error, warn};
use std::{sync::OnceLock, time::Instant};

mod ast_parser;
mod builtins;
mod compiler;
mod instructions;
mod tokenizer;
mod value;
mod vm;

static EXTRA_ARGS: OnceLock<Vec<String>> = OnceLock::new();

#[derive(Parser)]
struct Cli {
    /// Source file to execute
    source_file: String,

    /// Step through execution with enter (shows extra stuff if built with debug mode)
    #[clap(short, long)]
    step: bool,

    /// Print tokens with colors
    #[clap(short, long)]
    tokens: bool,

    /// 'debug' shows more compilation results, 'trace' shows instruction level details
    #[clap(short, long, value_parser = ["info", "debug", "trace"])]
    log_level: Option<String>,

    /// Extra arguments passed to the program
    #[arg(last = true)]
    extra_args: Vec<String>,
}

fn main() {
    let cli = Cli::parse();

    let mut log = env_logger::builder();
    log.format_timestamp(None);

    if let Some(log_level) = &cli.log_level {
        log.filter_level(match log_level.as_str() {
            "info" => log::LevelFilter::Info,
            "debug" => log::LevelFilter::Debug,
            "trace" => log::LevelFilter::Trace,
            _ => unreachable!(),
        });
    }

    if cli.step {
        log.filter_level(log::LevelFilter::Trace);
    }

    log.init();

    if !cfg!(debug_assertions) && cli.log_level.is_some_and(|lvl| lvl == "trace") {
        warn!(
            "Log level 'trace' selected, but it is available only in debug builds. Showing only 'debug' level logs."
        );
    }

    EXTRA_ARGS.set(cli.extra_args.to_vec()).unwrap();

    let source_code = match std::fs::read_to_string(&cli.source_file) {
        Ok(code) => code,
        Err(e) => {
            error!("Error reading source file '{}': {}", cli.source_file, e);
            std::process::exit(1);
        }
    };

    let bump = Bump::new();

    let tok_start = Instant::now();
    let tokens = tokenizer::tokenize(&bump, &source_code, cli.tokens);
    let tok_end = Instant::now();

    let parse_start = Instant::now();
    let block = ast_parser::parse(&bump, &source_code, &tokens);
    let parse_end = Instant::now();

    let compile_start = Instant::now();
    let compilation = compiler::compile(block, &source_code, &tokens);
    let compile_end = Instant::now();

    let exec_start = Instant::now();
    Vm::new(compilation).run(cli.step);
    let exec_end = Instant::now();

    println!(
        "tokenized in {:?}, parsed in {:?}, compiled in {:?}, executed in {:?}",
        tok_end - tok_start,
        parse_end - parse_start,
        compile_end - compile_start,
        exec_end - exec_start
    );
}
