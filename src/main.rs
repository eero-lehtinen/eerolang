use crate::{tokenizer::Token, vm::Vm};
use clap::Parser;
use std::{sync::OnceLock, time::Instant};

mod ast_parser;
mod builtins;
mod compiler;
mod instructions;
mod tokenizer;
mod value;
mod vm;

// Store these for convenient error reporting purposes.
static SOURCE: OnceLock<String> = OnceLock::new();
static TOKENS: OnceLock<Vec<Token>> = OnceLock::new();

#[derive(Parser)]
struct Cli {
    source_file: String,

    /// Step through execution with enter (shows extra stuff if built with debug mode)
    #[clap(long)]
    step: bool,

    /// Break at a specific line number (requires --step)
    #[clap(long, requires = "step")]
    break_at: Option<usize>,

    /// Print tokens with colors
    #[clap(long)]
    tokens: bool,
}

fn main() {
    let cli = Cli::parse();

    let mut log = env_logger::builder();
    log.format_timestamp(None);
    if cli.step {
        log.filter_level(log::LevelFilter::Trace);
    }
    log.init();

    let source_code = match std::fs::read_to_string(cli.source_file) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error reading source file: {}", e);
            std::process::exit(1);
        }
    };
    let source_code = SOURCE.get_or_init(|| source_code);

    let tok_start = Instant::now();
    let tokens = tokenizer::tokenize(source_code, cli.tokens);
    let tok_end = Instant::now();
    let tokens = TOKENS.get_or_init(|| tokens);

    let parse_start = Instant::now();
    let block = ast_parser::parse(tokens);
    let parse_end = Instant::now();

    let compile_start = Instant::now();
    let compilation = compiler::compile(&block, tokens);
    let compile_end = Instant::now();

    let exec_start = Instant::now();
    let stop_at_line = cli.step.then_some(cli.break_at.unwrap_or(0));
    Vm::new(compilation).run(stop_at_line);
    let exec_end = Instant::now();

    println!(
        "tokenized in {:?}, parsed in {:?}, compiled in {:?}, executed in {:?}",
        tok_end - tok_start,
        parse_end - parse_start,
        compile_end - compile_start,
        exec_end - exec_start
    );
}
