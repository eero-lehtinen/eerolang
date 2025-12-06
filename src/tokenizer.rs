use std::{cell::RefCell, rc::Rc};

use log::trace;

use crate::SOURCE;

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
pub struct Token {
    pub line: usize,
    pub byte_col: usize,
    pub byte_pos_start: usize,
    pub byte_pos_end: usize,
    pub index: usize,
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
    Range(Box<Range>),
}

pub fn dbg_display(values: &[Value]) -> String {
    if values.is_empty() {
        "[]".to_string()
    } else {
        let items = values
            .iter()
            .take(3)
            .map(|item| item.dbg_display())
            .collect::<Vec<String>>()
            .join(", ");
        if values.len() > 3 {
            format!("[{}, ...]", items)
        } else {
            format!("[{}]", items)
        }
    }
}

impl Value {
    pub fn dbg_display(&self) -> String {
        match self {
            Value::Integer(i) => format!("(int {})", i),
            Value::Float(f) => format!("(float {})", f),
            Value::String(s) => {
                let s = if s.len() <= 6 {
                    s.to_string()
                } else {
                    format!("{}...", &s[..6])
                };
                let s = s.replace("\n", "\\n");
                format!("(string \"{}\")", s)
            }
            Value::List(v) => {
                format!("(list {})", dbg_display(&v.borrow()))
            }
            Value::Range(r) => format!("(range {}, {})", r.start, r.end),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Range {
    pub start: i64,
    pub end: i64,
}

pub fn find_source_char_col(row: usize, byte_col: usize) -> usize {
    SOURCE
        .get()
        .unwrap()
        .lines()
        .nth(row)
        .and_then(|line| {
            line.char_indices()
                .enumerate()
                .find_map(|(char_idx, (bidx, _))| {
                    if bidx >= byte_col {
                        Some(char_idx)
                    } else {
                        None
                    }
                })
        })
        .unwrap_or(0)
}

pub fn report_source_pos(row: usize, char_col: usize) {
    for (i, src_line) in SOURCE.get().unwrap().lines().enumerate() {
        if (i as i64 - row as i64).abs() <= 2 {
            eprintln!("{:4} | {}", i + 1, src_line);
        }
        if i == row {
            eprintln!("       {}^", " ".repeat(char_col));
        }
    }
}

pub fn tokenize(source: &'_ str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut iter = source.char_indices().peekable();
    let mut tbuf = String::new();
    let mut row = 0;
    let mut byte_row_start = 0;

    macro_rules! panic_with_pos {
        ($msg:expr) => {
            let byte_col = iter.peek().map_or(0, |(i, _)| i - byte_row_start);
            let char_col = find_source_char_col(row, byte_col);
            eprintln!("{} at line {}, column {}:", &$msg, row + 1, char_col + 1);
            report_source_pos(row, byte_col);
            panic!("Tokenization failed");
        };
    }

    macro_rules! update_row {
        ($i:expr) => {
            row += 1;
            byte_row_start = $i + 1;
        };
    }

    while let Some((byte_pos, ch)) = iter.next() {
        macro_rules! tok {
            ($len:expr, $kind:expr) => {{
                let index = tokens.len();
                tokens.push(Token {
                    line: row,
                    byte_col: byte_pos - byte_row_start,
                    byte_pos_start: byte_pos,
                    byte_pos_end: byte_pos + $len,
                    index,
                    kind: $kind,
                });
            }};
        }

        match ch {
            '+' => tok!(1, TokenKind::Operator(Operator::Plus)),
            '-' => tok!(1, TokenKind::Operator(Operator::Minus)),
            '*' => tok!(1, TokenKind::Operator(Operator::Multiply)),
            '/' => tok!(1, TokenKind::Operator(Operator::Divide)),
            '<' => {
                if iter.peek().is_some_and(|(_, c)| *c == '=') {
                    iter.next();
                    tok!(2, TokenKind::Operator(Operator::Lte));
                } else {
                    tok!(1, TokenKind::Operator(Operator::Lt));
                }
            }
            '>' => {
                if iter.peek().is_some_and(|(_, c)| *c == '=') {
                    iter.next();
                    tok!(2, TokenKind::Operator(Operator::Gte));
                } else {
                    tok!(1, TokenKind::Operator(Operator::Gt));
                }
            }
            '!' => {
                if iter.peek().is_some_and(|(_, c)| *c == '=') {
                    iter.next();
                    tok!(2, TokenKind::Operator(Operator::Neq));
                } else {
                    panic!("Unexpected character: !");
                }
            }
            '=' => {
                if iter.peek().is_some_and(|(_, c)| *c == '=') {
                    iter.next();
                    tok!(2, TokenKind::Operator(Operator::Eq));
                } else {
                    tok!(1, TokenKind::Assign);
                }
            }
            '(' => tok!(1, TokenKind::LParen),
            ')' => tok!(1, TokenKind::RParen),
            '[' => tok!(1, TokenKind::LSquareParen),
            ']' => tok!(1, TokenKind::RSquareParen),
            '{' => tok!(1, TokenKind::LBrace),
            '}' => tok!(1, TokenKind::RBrace),
            ',' => tok!(1, TokenKind::Comma),
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
                    tbuf.len(),
                    TokenKind::Literal(Value::String(tbuf.clone().into()))
                );
                tbuf.clear();
            }
            ch if ch.is_alphabetic() || ch == '_' => {
                let mut byte_end_pos = byte_pos;
                while let Some(&(i, next_ch)) = iter.peek() {
                    byte_end_pos = i;
                    if next_ch.is_alphanumeric() || next_ch == '_' {
                        iter.next();
                    } else {
                        break;
                    }
                }
                match &source[byte_pos..byte_end_pos] {
                    "for" => tok!("for".len(), TokenKind::KeywordFor),
                    "in" => tok!("in".len(), TokenKind::KeywordIn),
                    "if" => tok!("if".len(), TokenKind::KeywordIf),
                    "else" => tok!("else".len(), TokenKind::KeywordElse),
                    ident => tok!(ident.len(), TokenKind::Ident(ident.to_string())),
                }
            }
            ch if ch.is_ascii_digit() => {
                let mut byte_end_pos = byte_pos;
                let mut is_float = false;
                while let Some(&(i, next_ch)) = iter.peek() {
                    byte_end_pos = i;
                    if next_ch.is_ascii_digit() {
                        iter.next();
                    } else if next_ch == '.' && !is_float {
                        is_float = true;
                        iter.next();
                    } else {
                        break;
                    }
                }
                if iter
                    .peek()
                    .is_some_and(|(_, ch)| ch.is_ascii_alphanumeric() || *ch == '_')
                {
                    panic_with_pos!(format!(
                        "Invalid numeric literal: '{}{}'",
                        &source[byte_pos..byte_end_pos],
                        iter.peek().unwrap().1
                    ));
                }
                let data = &source[byte_pos..byte_end_pos];
                if is_float {
                    if let Ok(float_val) = data.parse::<f64>() {
                        tok!(data.len(), TokenKind::Literal(Value::Float(float_val)));
                    } else {
                        panic_with_pos!(format!("Invalid float literal: '{}'", data));
                    }
                } else if let Ok(int_val) = data.parse::<i64>() {
                    tok!(data.len(), TokenKind::Literal(Value::Integer(int_val)));
                } else {
                    panic_with_pos!(format!("Invalid integer literal: '{}'", data));
                }
            }
            '\n' => {
                update_row!(byte_pos);
            }
            ch if ch.is_whitespace() => {}
            _ => {
                panic_with_pos!(format!("Unexpected character: {}", ch));
            }
        }
    }

    trace!("Tokenized source:\n{:#?}", tokens);

    tokens
}
