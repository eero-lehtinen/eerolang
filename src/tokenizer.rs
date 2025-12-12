use std::{
    fmt::Display,
    io::{StdoutLock, Write},
    ops::Range,
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
                eprintln!("{} at line {}, column {}:", &$msg, row + 1, char_col + 1);
                let context = 2;
                report_source_pos(&tokens, row, byte_col, byte_pos, byte_pos_end, context);
                eprintln!("Tokenization failed");
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
                // if is_float {
                if let Ok(float_val) = data.parse::<f64>() {
                    tok!(data.len(), TokenKind::Literal(Literal::Number(float_val)));
                } else {
                    panic_with_pos!(format!("Invalid number literal: '{}'", data));
                }
                // } else if let Ok(int_val) = data.parse::<i64>() {
                //     tok!(data.len(), TokenKind::Literal(Value::int(int_val)));
                // } else {
                //     panic_with_pos!(format!("Invalid integer literal: '{}'", data));
                // }
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
        print_colored_tokens(&tokens, None, None);
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
) {
    print_colored_tokens(
        tokens,
        Some(row.saturating_sub(context as usize)..(row + context as usize + 1)),
        Some((row, char_col, byte_pos_start, byte_pos_end)),
    );
}

fn print_colored_tokens(
    tokens: &[Token],
    line_range: Option<Range<usize>>,
    error_pos: Option<(usize, usize, usize, usize)>,
) {
    let source = SOURCE.get().unwrap();
    let mut stdout = std::io::stdout().lock();

    let line_start = |stdout: &mut StdoutLock, line: usize| {
        write!(
            stdout,
            "{}",
            format!("{:4} | ", line + 1)
                .color(colored::Color::BrightBlack)
                .on_color(colored::Color::Black),
        )
        .unwrap()
    };

    let show_error = |stdout: &mut StdoutLock, line: usize| {
        if let Some((err_line, err_col, _, _)) = error_pos
            && line == err_line
        {
            writeln!(
                stdout,
                "{}{}",
                " ".repeat(err_col + 7),
                "^".color(colored::Color::Red)
            )
            .unwrap();
        }
    };

    let mut tokens = tokens
        .iter()
        .filter(|t| {
            if let Some(range) = &line_range {
                t.line >= range.start && t.line < range.end
            } else {
                true
            }
        })
        .peekable();

    let mut line = 0;

    if let Some(range) = &line_range {
        line = range.start;
    }

    let mut byte_pos = tokens.peek().map_or(0, |t| t.byte_pos_start - t.byte_col);

    line_start(&mut stdout, line);
    for tok in tokens {
        while byte_pos < tok.byte_pos_start {
            let ch = source.as_bytes()[byte_pos] as char;
            if ch == '\n' {
                line += 1;
                writeln!(stdout).unwrap();

                show_error(&mut stdout, line);
                line_start(&mut stdout, line);
            } else {
                let non_tok = &source[byte_pos..byte_pos + 1].color(colored::Color::Red);
                write!(stdout, "{}", non_tok).unwrap();
            }
            byte_pos += 1;
        }
        let color = if error_pos.is_some_and(|(_, _, err_byte_pos_start, err_byte_pos_end)| {
            tok.byte_pos_start == err_byte_pos_start && tok.byte_pos_end == err_byte_pos_end
        }) {
            colored::Color::Red
        } else {
            tok.kind.color()
        };
        let c = source[tok.byte_pos_start..tok.byte_pos_end].color(color);
        byte_pos = tok.byte_pos_end;
        write!(stdout, "{}", c).unwrap();
    }

    // If errored while tokenizing, the erroring token won't exist, so we have to highlight normal
    // text here.
    if let Some((err_line, _, err_byte_pos_start, err_byte_pos_end)) = error_pos {
        while let Some(ch) = source.as_bytes().get(byte_pos).cloned() {
            let ch = ch as char;
            if ch == '\n' {
                line += 1;
                writeln!(stdout).unwrap();

                show_error(&mut stdout, line);
                if line > err_line {
                    break;
                }
                line_start(&mut stdout, line);
            } else {
                let color = if byte_pos >= err_byte_pos_start && byte_pos < err_byte_pos_end {
                    colored::Color::Red
                } else {
                    colored::Color::White
                };
                let non_tok = &source[byte_pos..byte_pos + 1].color(color);
                write!(stdout, "{}", non_tok).unwrap();
            }
            byte_pos += 1;

            if byte_pos >= err_byte_pos_end && line >= err_line {
                break;
            }
        }
    }

    writeln!(&mut stdout).unwrap();
}
