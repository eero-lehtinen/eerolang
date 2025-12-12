use std::{
    fmt::Display,
    io::{StderrLock, Write},
};

use colored::Colorize;

use crate::SOURCE;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    DeclareAssign,
    Assign,
    Operator(Operator),
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Literal(Literal),
    Ident(String),
    KeywordFor,
    KeywordWhile,
    KeywordIn,
    KeywordIf,
    KeywordElse,
    KeywordContinue,
    KeywordBreak,
    KeywordFn,
    KeywordReturn,
    Comment,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Number(f64),
    String(String),
}

impl TokenKind {
    pub fn color(&self) -> colored::Color {
        match self {
            TokenKind::DeclareAssign | TokenKind::Assign | TokenKind::Operator(_) => {
                colored::Color::Yellow
            }
            TokenKind::LParen
            | TokenKind::RParen
            | TokenKind::LBrace
            | TokenKind::RBrace
            | TokenKind::Comma => colored::Color::White,
            TokenKind::Literal(Literal::String(_)) => colored::Color::Green,
            TokenKind::Literal(Literal::Number(_)) => colored::Color::BrightCyan,
            TokenKind::Ident(_) => colored::Color::Cyan,
            TokenKind::KeywordFor
            | TokenKind::KeywordWhile
            | TokenKind::KeywordIn
            | TokenKind::KeywordIf
            | TokenKind::KeywordElse
            | TokenKind::KeywordContinue
            | TokenKind::KeywordBreak
            | TokenKind::KeywordFn
            | TokenKind::KeywordReturn => colored::Color::Magenta,
            TokenKind::Comment => colored::Color::BrightBlack,
        }
    }
}

impl Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenKind::DeclareAssign => write!(f, ":="),
            TokenKind::Assign => write!(f, "="),
            TokenKind::Operator(op) => write!(f, "{}", op),
            TokenKind::LParen => write!(f, "("),
            TokenKind::RParen => write!(f, ")"),
            TokenKind::LBrace => write!(f, "{{"),
            TokenKind::RBrace => write!(f, "}}"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Literal(val) => write!(f, "{:?}", val),
            TokenKind::Ident(name) => write!(f, "ident({})", name),
            TokenKind::KeywordFor => write!(f, "for"),
            TokenKind::KeywordWhile => write!(f, "while"),
            TokenKind::KeywordIn => write!(f, "in"),
            TokenKind::KeywordIf => write!(f, "if"),
            TokenKind::KeywordElse => write!(f, "else"),
            TokenKind::KeywordContinue => write!(f, "continue"),
            TokenKind::KeywordBreak => write!(f, "break"),
            TokenKind::KeywordFn => write!(f, "fn"),
            TokenKind::KeywordReturn => write!(f, "return"),
            TokenKind::Comment => write!(f, "# <comment>"),
        }
    }
}

#[derive(Debug, Clone)]
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
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
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
        };
        write!(f, "{}", text)
    }
}

pub fn tokenize(source: &'_ str, show: bool) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut iter = source.char_indices().peekable();
    let mut tbuf = String::new();
    let mut row = 0;
    let mut byte_row_start = 0;

    macro_rules! update_row {
        ($i:expr) => {
            row += 1;
            byte_row_start = $i + 1;
        };
    }

    while let Some((byte_pos, ch)) = iter.next() {
        macro_rules! panic_with_pos {
            ($msg:expr) => {
                let byte_pos_end = iter.peek().map_or(byte_pos + 1, |(i, _)| *i + 1);
                let byte_col = byte_pos - byte_row_start;
                let char_col = find_source_char_col(row, byte_col);
                eprintln!("Tokenization failed");
                eprintln!("{} at line {}, column {}:", &$msg, row + 1, char_col + 1);
                let context = 2;
                report_source_pos(
                    &tokens,
                    row,
                    byte_col,
                    byte_pos,
                    byte_pos_end,
                    context,
                    colored::Color::BrightRed,
                );
                std::process::exit(1);
            };
        }
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
                    panic_with_pos!(format!("Expected '=' after '{}'", ch));
                }
            }
            ':' => {
                if iter.peek().is_some_and(|(_, c)| *c == '=') {
                    iter.next();
                    tok!(2, TokenKind::DeclareAssign);
                } else {
                    panic_with_pos!(format!("Expected '=' after '{}'", ch));
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
            '{' => tok!(1, TokenKind::LBrace),
            '}' => tok!(1, TokenKind::RBrace),
            ',' => tok!(1, TokenKind::Comma),
            '#' => {
                let start = byte_pos + 1;
                for (i, next_ch) in iter.by_ref() {
                    if next_ch == '\n' {
                        tok!(i - start + 1, TokenKind::Comment);
                        update_row!(i);
                        break;
                    }
                }
            }
            '"' => {
                tbuf.clear();
                let mut escape = false;
                let mut tok_len = 0;
                for (_, next_ch) in iter.by_ref() {
                    tok_len += next_ch.len_utf8();
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
                    tok_len + 1,
                    TokenKind::Literal(Literal::String(tbuf.clone()))
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
                    "while" => tok!("while".len(), TokenKind::KeywordWhile),
                    "in" => tok!("in".len(), TokenKind::KeywordIn),
                    "if" => tok!("if".len(), TokenKind::KeywordIf),
                    "else" => tok!("else".len(), TokenKind::KeywordElse),
                    "continue" => tok!("continue".len(), TokenKind::KeywordContinue),
                    "break" => tok!("break".len(), TokenKind::KeywordBreak),
                    "fn" => tok!("fn".len(), TokenKind::KeywordFn),
                    "return" => tok!("return".len(), TokenKind::KeywordReturn),
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
                if let Ok(float_val) = data.parse::<f64>() {
                    tok!(data.len(), TokenKind::Literal(Literal::Number(float_val)));
                } else {
                    panic_with_pos!(format!("Invalid number literal: '{}'", data));
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

    if show {
        print_colored_tokens(&tokens, None);
    }

    tokens
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

pub fn report_source_pos(
    tokens: &[Token],
    row: usize,
    char_col: usize,
    byte_pos_start: usize,
    byte_pos_end: usize,
    context: u32,
    color: colored::Color,
) {
    print_colored_tokens(
        tokens,
        Some((
            row,
            context as usize,
            char_col,
            byte_pos_start,
            byte_pos_end,
            color,
        )),
    );
}

fn print_colored_tokens(
    tokens: &[Token],
    highlight: Option<(usize, usize, usize, usize, usize, colored::Color)>,
) {
    let source = SOURCE.get().unwrap();
    let mut stderr = std::io::stderr().lock();

    let line_start = |out: &mut StderrLock, line: usize| {
        write!(
            out,
            "{}",
            format!("{:4} | ", line + 1)
                .color(colored::Color::BrightBlack)
                .on_color(colored::Color::Black),
        )
        .unwrap()
    };

    let show_hl = |out: &mut StderrLock, line: usize| {
        if let Some((err_line, _, err_col, byte_pos_start, byte_pos_end, color)) = highlight
            && line == err_line
        {
            writeln!(
                out,
                "{}{}{}",
                " ".repeat(err_col + 7),
                "^".color(color),
                "~".repeat((byte_pos_end - byte_pos_start).saturating_sub(1))
                    .color(color)
            )
            .unwrap();
        }
    };

    let mut token_iter = tokens.iter().peekable();

    let mut line = 0;

    let mut byte_pos = 0;
    let bytes = source.as_bytes();

    if let Some((row, context, ..)) = &highlight {
        while byte_pos < source.len() {
            let ch = bytes[byte_pos] as char;
            if ch == '\n' {
                if line + 1 >= row.saturating_sub(*context) {
                    break;
                }
                line += 1;
            }
            byte_pos += 1;
        }
    }

    let hl_color = |byte_pos: usize, tok: Option<&&Token>| {
        let byte_pos = tok.map(|t| t.byte_pos_start).unwrap_or(byte_pos);
        highlight.and_then(|(_, _, _, err_byte_pos_start, err_byte_pos_end, color)| {
            if byte_pos >= err_byte_pos_start && byte_pos < err_byte_pos_end {
                Some(color)
            } else {
                None
            }
        })
    };

    if line == 0 {
        line_start(&mut stderr, line);
    }
    while byte_pos < source.len() {
        while let Some(tok) = token_iter.peek()
            && byte_pos >= tok.byte_pos_end
        {
            token_iter.next();
        }

        let ch = bytes[byte_pos] as char;
        if ch == '\n' {
            writeln!(stderr).unwrap();

            if let Some((row, context, ..)) = highlight
                && (line + 1) > row + context
            {
                break;
            }

            show_hl(&mut stderr, line);
            line += 1;
            line_start(&mut stderr, line);
        } else {
            let tok = token_iter.peek();
            let color = hl_color(byte_pos, tok)
                .unwrap_or_else(|| tok.map(|t| t.kind.color()).unwrap_or(colored::Color::White));
            let text = &source[byte_pos..byte_pos + 1].color(color);
            write!(stderr, "{}", text).unwrap();
        }

        byte_pos += 1;
    }

    writeln!(&mut stderr).unwrap();
}
