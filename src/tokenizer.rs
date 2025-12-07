use std::{cell::RefCell, fmt::Display, rc::Rc};

use foldhash::HashMap;
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

impl Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenKind::Assign => write!(f, "="),
            TokenKind::Operator(op) => write!(f, "{}", op.dbg_display()),
            TokenKind::LParen => write!(f, "("),
            TokenKind::RParen => write!(f, ")"),
            TokenKind::LSquareParen => write!(f, "["),
            TokenKind::RSquareParen => write!(f, "]"),
            TokenKind::LBrace => write!(f, "{{"),
            TokenKind::RBrace => write!(f, "}}"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Literal(val) => write!(f, "{}", val.dbg_display()),
            TokenKind::Ident(name) => write!(f, "ident({})", name),
            TokenKind::KeywordFor => write!(f, "for"),
            TokenKind::KeywordIn => write!(f, "in"),
            TokenKind::KeywordIf => write!(f, "if"),
            TokenKind::KeywordElse => write!(f, "else"),
        }
    }
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
    Add,
    Sub,
    Mul,
    Div,
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
            Operator::Add | Operator::Sub => 1,
            Operator::Mul | Operator::Div => 2,
        }
    }

    pub fn dbg_display(&self) -> &'static str {
        match self {
            Operator::Add => "+",
            Operator::Sub => "-",
            Operator::Mul => "*",
            Operator::Div => "/",
            Operator::Lt => "<",
            Operator::Gt => ">",
            Operator::Lte => "<=",
            Operator::Gte => ">=",
            Operator::Eq => "==",
            Operator::Neq => "!=",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Integer(i64),
    Float(f64),
    String(Rc<String>),
    List(Rc<RefCell<Vec<Value>>>),
    Map(Rc<RefCell<HashMap<String, Value>>>),
    Range(Box<Range>),
}

impl Value {
    pub fn dbg_display(&self) -> String {
        match self {
            Value::Integer(i) => format!("int {}", i),
            Value::Float(f) => format!("float {}", f),
            Value::String(s) => {
                let s = if s.len() <= 6 {
                    s.to_string()
                } else {
                    format!("{}...", &s[..6])
                };
                let s = s.replace("\n", "\\n");
                format!("string \"{}\"", s)
            }
            Value::List(v) => {
                format!("list {}", dbg_display(&v.borrow()))
            }
            Value::Map(m) => {
                let map = m.borrow();
                let items = map
                    .iter()
                    .take(2)
                    .map(|(k, v)| format!("{}: {}", k, v.dbg_display()))
                    .collect::<Vec<String>>()
                    .join(", ");
                if map.len() > 2 {
                    format!("map {{{}, ...}}", items)
                } else {
                    format!("map {{{}}}", items)
                }
            }
            Value::Range(r) => format!("range {}, {}", r.start, r.end),
        }
    }

    #[inline]
    pub fn float_promoted(&self) -> Option<f64> {
        Some(match self {
            Value::Integer(i) => *i as f64,
            Value::Float(f) => *f,
            _ => return None,
        })
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Value::Integer(value)
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::Integer(i64::from(value))
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::Float(value)
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::String(Rc::new(value))
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Value::String(Rc::new(value.to_string()))
    }
}

impl From<Vec<Value>> for Value {
    fn from(value: Vec<Value>) -> Self {
        Value::List(Rc::new(RefCell::new(value)))
    }
}

pub fn arg_display(values: &[Value]) -> String {
    let items = values
        .iter()
        .map(|item| item.dbg_display())
        .collect::<Vec<String>>()
        .join(", ");
    format!("({})", items)
}

pub fn dbg_display(values: &[Value]) -> String {
    if values.is_empty() {
        "[]".to_string()
    } else {
        let items = values
            .iter()
            .take(2)
            .map(|item| item.dbg_display())
            .collect::<Vec<String>>()
            .join(", ");
        if values.len() > 2 {
            format!("[{}, ...]", items)
        } else {
            format!("[{}]", items)
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
            '+' => tok!(1, TokenKind::Operator(Operator::Add)),
            '-' => tok!(1, TokenKind::Operator(Operator::Sub)),
            '*' => tok!(1, TokenKind::Operator(Operator::Mul)),
            '/' => tok!(1, TokenKind::Operator(Operator::Div)),
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
