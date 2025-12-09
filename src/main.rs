use crate::vm::Vm;
use clap::Parser;
use log::trace;
use std::{sync::OnceLock, time::Instant};

mod ast_parser;
mod builtins;
mod compiler;
mod tokenizer;
mod vm;
mod value;



static SOURCE: OnceLock<String> = OnceLock::new();
#[derive(Parser)]
struct Cli {
    source_file: String,

    /// Step through execution with enter (shows extra stuff if built with debug mode)
    #[clap(short, long)]
    step: bool,

    /// Print tokens with colors
    #[clap(short, long)]
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
    SOURCE.set(source_code.clone()).unwrap();

    let tok_start = Instant::now();
    let tokens = tokenizer::tokenize(&source_code, cli.tokens);
    let tok_end = Instant::now();
    for token in &tokens {
        trace!("{:?}", token);
    }

    let parse_start = Instant::now();
    let block = ast_parser::parse(&tokens);
    let parse_end = Instant::now();

    let compile_start = Instant::now();
    let compilation = compiler::compile(&block, &tokens);
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
