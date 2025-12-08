use std::{sync::OnceLock, time::Instant};

use log::{error, trace};

use crate::vm::Vm;

mod ast_parser;
mod builtins;
mod compiler;
mod tokenizer;
mod vm;

static SOURCE: OnceLock<String> = OnceLock::new();

fn main() {
    env_logger::builder().format_timestamp(None).init();

    let args = std::env::args().collect::<Vec<String>>();
    if args.len() < 2 {
        error!("Usage: {} <source_file>", args[0]);
        return;
    }

    let source_file = &args[1];
    let source_code = std::fs::read_to_string(source_file).expect("Failed to read source file");
    SOURCE.set(source_code.clone()).unwrap();

    let tok_start = Instant::now();
    let tokens = tokenizer::tokenize(&source_code);
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
    Vm::new(compilation).run();
    let exec_end = Instant::now();

    println!(
        "tokenized in {:?}, parsed in {:?}, compiled in {:?}, executed in {:?}",
        tok_end - tok_start,
        parse_end - parse_start,
        compile_end - compile_start,
        exec_end - exec_start
    );
}
