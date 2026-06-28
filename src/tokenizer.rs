use std::{
    fmt::Display,
    io::{StderrLock, Write},
};

use bumpalo::Bump;
use colored::Colorize;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind<'a> {
    DeclareAssign,
    Assign,
    Operator(Operator),
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Literal(Literal<'a>),
    Ident(&'a str),
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
pub enum Literal<'a> {
    Number(f64),
    String(&'a str),
    Null,
}

impl TokenKind<'_> {
    pub fn color(&self) -> colored::Color {
        match self {
            TokenKind::DeclareAssign | TokenKind::Assign | TokenKind::Operator(_) => {
                colored::Color::Yellow
            }
            TokenKind::LParen
            | TokenKind::RParen
            | TokenKind::LBrace
            | TokenKind::RBrace
            | TokenKind::LBracket
            | TokenKind::RBracket
            | TokenKind::Comma => colored::Color::White,
            TokenKind::Literal(Literal::String(_)) => colored::Color::Green,
            TokenKind::Literal(Literal::Number(_) | Literal::Null) => colored::Color::BrightCyan,
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

impl Display for TokenKind<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenKind::DeclareAssign => write!(f, ":="),
            TokenKind::Assign => write!(f, "="),
            TokenKind::Operator(op) => write!(f, "{}", op),
            TokenKind::LParen => write!(f, "("),
            TokenKind::RParen => write!(f, ")"),
            TokenKind::LBrace => write!(f, "{{"),
            TokenKind::RBrace => write!(f, "}}"),
            TokenKind::LBracket => write!(f, "["),
            TokenKind::RBracket => write!(f, "]"),
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
pub struct Token<'a> {
    pub line: usize,
    pub byte_col: usize,
    pub byte_pos_start: usize,
    pub byte_pos_end: usize,
    pub index: usize,
    pub kind: TokenKind<'a>,
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
    And,
    Or,
}

impl Operator {
    pub fn precedence(&self) -> u8 {
        match self {
            Operator::Or => 0,
            Operator::And => 1,
            Operator::Lt
            | Operator::Gt
            | Operator::Lte
            | Operator::Gte
            | Operator::Eq
            | Operator::Neq => 2,
            Operator::Add | Operator::Sub => 3,
            Operator::Mul | Operator::Div => 4,
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
            Operator::And => "and",
            Operator::Or => "or",
        };
        write!(f, "{}", text)
    }
}

pub fn tokenize<'a>(bump: &'a Bump, source: &'a str, show: bool) -> Vec<Token<'a>> {
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
                let char_col = find_source_char_col(source, row, byte_col);
                eprintln!("Tokenization failed");
                eprintln!("{} at line {}, column {}:", &$msg, row + 1, char_col + 1);
                let context = 2;
                report_source(
                    source,
                    &tokens,
                    Some((
                        SourcePos {
                            row,
                            char_col: byte_col,
                            byte_pos_start: byte_pos,
                            byte_pos_end,
                        },
                        context as usize,
                        colored::Color::BrightRed,
                    )),
                );
                std::panic::panic_any(crate::LangError(format!(
                    "{} at line {}, column {}",
                    $msg,
                    row + 1,
                    char_col + 1
                )));
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
            '[' => tok!(1, TokenKind::LBracket),
            ']' => tok!(1, TokenKind::RBracket),
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
                let mut has_escapes = false;
                let mut tok_len = 0;
                let mut closed = false;
                let str_content_start = byte_pos + 1; // after the opening quote
                for (_, next_ch) in iter.by_ref() {
                    tok_len += next_ch.len_utf8();
                    if next_ch == '\\' && !escape {
                        escape = true;
                        has_escapes = true;
                        continue;
                    }
                    if next_ch == '"' && !escape {
                        closed = true;
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
                if !closed {
                    panic_with_pos!("Unterminated string literal");
                }
                let str_val: &'a str = if has_escapes {
                    bump.alloc_str(&tbuf)
                } else {
                    // No escapes: borrow directly from source (content between quotes)
                    let str_content_end = str_content_start + tok_len - 1; // before the closing quote
                    &source[str_content_start..str_content_end]
                };
                tok!(tok_len + 1, TokenKind::Literal(Literal::String(str_val)));
                tbuf.clear();
            }
            ch if ch.is_alphabetic() || ch == '_' => {
                let mut byte_end_pos = source.len();
                while let Some(&(i, next_ch)) = iter.peek() {
                    if next_ch.is_alphanumeric() || next_ch == '_' {
                        iter.next();
                    } else {
                        byte_end_pos = i;
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
                    "and" => tok!("and".len(), TokenKind::Operator(Operator::And)),
                    "or" => tok!("or".len(), TokenKind::Operator(Operator::Or)),
                    "null" => tok!("null".len(), TokenKind::Literal(Literal::Null)),
                    ident => tok!(ident.len(), TokenKind::Ident(ident)),
                }
            }
            ch if ch.is_ascii_digit() => {
                let mut byte_end_pos = source.len();
                let mut is_float = false;
                while let Some(&(i, next_ch)) = iter.peek() {
                    if next_ch.is_ascii_digit() {
                        iter.next();
                    } else if next_ch == '.' && !is_float {
                        is_float = true;
                        iter.next();
                    } else {
                        byte_end_pos = i;
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
        report_source(source, &tokens, None);
    }

    tokens
}

pub fn find_source_char_col(source: &str, row: usize, byte_col: usize) -> usize {
    source
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

#[derive(Debug, Clone, Copy)]
pub struct SourcePos {
    pub row: usize,
    pub char_col: usize,
    pub byte_pos_start: usize,
    pub byte_pos_end: usize,
}

pub fn report_source(
    source: &str,
    tokens: &[Token<'_>],
    highlight: Option<(SourcePos, usize, colored::Color)>,
) {
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
        if let Some((pos, _, color)) = highlight
            && line == pos.row
        {
            writeln!(
                out,
                "{}{}{}",
                " ".repeat(pos.char_col + 7),
                "^".color(color),
                "~".repeat((pos.byte_pos_end - pos.byte_pos_start).saturating_sub(1))
                    .color(color)
            )
            .unwrap();
        }
    };

    let mut token_iter = tokens.iter().peekable();

    let mut line = 0;

    let mut byte_pos = 0;
    let bytes = source.as_bytes();

    if let Some((pos, context, ..)) = &highlight
        && pos.row.saturating_sub(*context) > 0
    {
        while byte_pos < source.len() {
            let ch = bytes[byte_pos] as char;
            if ch == '\n' {
                if line + 1 >= pos.row.saturating_sub(*context) {
                    break;
                }
                line += 1;
            }
            byte_pos += 1;
        }
    }

    let hl_color = |byte_pos: usize, tok: Option<&&Token<'_>>| {
        let byte_pos = tok.map(|t| t.byte_pos_start).unwrap_or(byte_pos);
        highlight.and_then(|(pos, _, color)| {
            if byte_pos >= pos.byte_pos_start && byte_pos < pos.byte_pos_end {
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

            if let Some((pos, context, ..)) = highlight
                && (line + 1) > pos.row + context
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
