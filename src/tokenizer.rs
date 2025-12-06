use std::{cell::RefCell, rc::Rc};

use log::trace;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Assign,
    Operator(Operator),
    LParen,
    RParen,
    LSquareParen,
    RSquareParen,
    LBrace,
    RBrace,
    Comma,
    Literal(Value),
    Ident(String),
    KeywordFor,
    KeywordIn,
    KeywordIf,
    KeywordElse,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token<'a> {
    pub line: usize,
    pub column: usize,
    pub text: &'a str,
    pub kind: TokenKind,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Operator {
    Plus,
    Minus,
    Multiply,
    Divide,
    Lt,
    Gt,
    Lte,
    Gte,
    Eq,
    Neq,
}

impl Operator {
    pub fn precedence(&self) -> u8 {
        match self {
            Operator::Lt
            | Operator::Gt
            | Operator::Lte
            | Operator::Gte
            | Operator::Eq
            | Operator::Neq => 0,
            Operator::Plus | Operator::Minus => 1,
            Operator::Multiply | Operator::Divide => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Integer(i64),
    Float(f64),
    String(Rc<str>),
    List(Rc<RefCell<Vec<Value>>>),
    Range(i64, i64),
}

fn report_source_pos(source: &str, row: usize, col: usize, err: &str) {
    eprintln!("{} at line {}, column {}:", err, row + 1, col);
    for (i, src_line) in source.lines().enumerate() {
        if (i as i64 - row as i64).abs() <= 2 {
            eprintln!("{:4} | {}", i + 1, src_line);
        }
        if i == row {
            eprintln!("       {}^", " ".repeat(col.saturating_sub(1)));
        }
    }
}

pub fn tokenize(source: &'_ str) -> Vec<Token<'_>> {
    let mut tokens = Vec::new();
    let mut iter = source.chars().enumerate().peekable();
    let mut tbuf = String::new();
    let mut row = 0;
    let mut row_start = 0;

    macro_rules! panic_with_pos {
        ($msg:expr) => {
            let col = iter.peek().map_or(0, |(i, _)| i - row_start);
            report_source_pos(source, row, col, &$msg);
            panic!("Tokenization failed");
        };
    }

    macro_rules! update_row {
        ($i:expr) => {
            row += 1;
            row_start = $i + 1;
        };
    }

    while let Some((i, ch)) = iter.next() {
        macro_rules! tok {
            ($text:expr, $kind:expr) => {
                tokens.push(Token {
                    line: row,
                    column: i - row_start,
                    text: $text,
                    kind: $kind,
                })
            };
        }

        match ch {
            '+' => tok!("+", TokenKind::Operator(Operator::Plus)),
            '-' => tok!("-", TokenKind::Operator(Operator::Minus)),
            '*' => tok!("*", TokenKind::Operator(Operator::Multiply)),
            '/' => tok!("/", TokenKind::Operator(Operator::Divide)),
            '<' => {
                if iter.peek().is_some_and(|(_, c)| *c == '=') {
                    iter.next();
                    tok!("<=", TokenKind::Operator(Operator::Lte));
                } else {
                    tok!("<", TokenKind::Operator(Operator::Lt));
                }
            }
            '>' => {
                if iter.peek().is_some_and(|(_, c)| *c == '=') {
                    iter.next();
                    tok!(">=", TokenKind::Operator(Operator::Gte));
                } else {
                    tok!(">", TokenKind::Operator(Operator::Gt));
                }
            }
            '!' => {
                if iter.peek().is_some_and(|(_, c)| *c == '=') {
                    iter.next();
                    tok!("!=", TokenKind::Operator(Operator::Neq));
                } else {
                    panic!("Unexpected character: !");
                }
            }
            '=' => {
                if iter.peek().is_some_and(|(_, c)| *c == '=') {
                    iter.next();
                    tok!("==", TokenKind::Operator(Operator::Eq));
                } else {
                    tok!("=", TokenKind::Assign);
                }
            }
            '(' => tok!("(", TokenKind::LParen),
            ')' => tok!(")", TokenKind::RParen),
            '[' => tok!("[", TokenKind::LSquareParen),
            ']' => tok!("]", TokenKind::RSquareParen),
            '{' => tok!("{", TokenKind::LBrace),
            '}' => tok!("}", TokenKind::RBrace),
            ',' => tok!(",", TokenKind::Comma),
            '#' => {
                for (i, next_ch) in iter.by_ref() {
                    if next_ch == '\n' {
                        update_row!(i);
                        break;
                    }
                }
            }
            '"' => {
                tbuf.clear();
                let mut escape = false;
                for (_, next_ch) in iter.by_ref() {
                    if next_ch == '\\' && !escape {
                        escape = true;
                        continue;
                    }
                    if next_ch == '"' && !escape {
                        break;
                    }
                    if escape {
                        match next_ch {
                            'n' => tbuf.push('\n'),
                            't' => tbuf.push('\t'),
                            'r' => tbuf.push('\r'),
                            '\\' => tbuf.push('\\'),
                            '"' => tbuf.push('"'),
                            other => tbuf.push(other),
                        }
                    } else {
                        tbuf.push(next_ch);
                    }
                    escape = false;
                }
                tok!(
                    &source[i..i + tbuf.len() + 1],
                    TokenKind::Literal(Value::String(tbuf.clone().into()))
                );
                tbuf.clear();
            }
            ch if ch.is_alphabetic() || ch == '_' => {
                let mut j = i;
                while let Some(&(_, next_ch)) = iter.peek() {
                    j += 1;
                    if next_ch.is_alphanumeric() || next_ch == '_' {
                        iter.next();
                    } else {
                        break;
                    }
                }
                match &source[i..j] {
                    "for" => tok!("for", TokenKind::KeywordFor),
                    "in" => tok!("in", TokenKind::KeywordIn),
                    "if" => tok!("if", TokenKind::KeywordIf),
                    "else" => tok!("else", TokenKind::KeywordElse),
                    ident => tok!(ident, TokenKind::Ident(ident.to_string())),
                }
            }
            ch if ch.is_ascii_digit() => {
                let mut j = i;
                let mut is_float = false;
                while let Some(&(_, next_ch)) = iter.peek() {
                    j += 1;
                    if next_ch.is_ascii_digit() {
                        iter.next();
                    } else if next_ch == '.' && !is_float {
                        is_float = true;
                        iter.next();
                    } else {
                        break;
                    }
                }
                let data = &source[i..j];
                if is_float {
                    if let Ok(float_val) = data.parse::<f64>() {
                        tok!(data, TokenKind::Literal(Value::Float(float_val)));
                    } else {
                        panic_with_pos!(format!("Invalid float literal: '{}'", data));
                    }
                } else if let Ok(int_val) = data.parse::<i64>() {
                    tok!(data, TokenKind::Literal(Value::Integer(int_val)));
                } else {
                    panic_with_pos!(format!("Invalid integer literal: '{}'", data));
                }
            }
            ch if ch.is_whitespace() => {
                if ch == '\n' {
                    update_row!(i);
                }
            }
            _ => {
                panic_with_pos!(format!("Unexpected character: {}", ch));
            }
        }
    }

    trace!("Tokenized source:\n{:#?}", tokens);

    tokens
}
