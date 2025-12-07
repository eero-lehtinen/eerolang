use std::iter::Peekable;

use log::trace;

use crate::{
    SOURCE,
    tokenizer::{Operator, Token, TokenKind, Value, find_source_char_col, report_source_pos},
};

type VarName = String;

#[derive(Debug)]
pub enum AstNodeKind {
    Assign(VarName, Box<AstNode>),
    FunctionCall(VarName, Vec<AstNode>),
    /// index, key, item, list, block
    ForLoop(
        VarName,
        VarName,
        Option<VarName>,
        Box<AstNode>,
        Box<AstNode>,
    ),
    Continue,
    Break,
    /// condition, then block, else block
    IfStatement(Box<AstNode>, Box<AstNode>, Option<Box<AstNode>>),
    BinaryOp(Box<AstNode>, Operator, Box<AstNode>),
    Block(Vec<AstNode>, Vec<String>),
    Literal(Value),
    Variable(VarName),
}

#[derive(Debug)]
pub struct AstNode {
    pub token_idx: usize,
    pub kind: AstNodeKind,
}

trait TokIter<'a>: Iterator<Item = &'a Token> + Clone {}
impl<'a, T: Iterator<Item = &'a Token> + Clone> TokIter<'a> for T {}

fn parse_list<'a, I: TokIter<'a>>(
    iter: &mut Peekable<I>,
    separator: TokenKind,
    end_token: TokenKind,
) -> Vec<AstNode> {
    let mut elements = Vec::new();
    loop {
        let next = iter.peek();
        if next.is_some_and(|t| t.kind == end_token) {
            iter.next();
            break;
        }
        let element = parse_expression(iter).unwrap_or_else(|| {
            fatal("Expected expression in list", iter.peek().unwrap());
        });
        elements.push(element);
        let next = iter.peek();
        if next.is_some_and(|t| t.kind == separator) {
            iter.next();
        } else if next.is_some_and(|t| t.kind == end_token) {
            iter.next();
            break;
        } else {
            fatal(
                &format!(
                    "Expected separator '{}' or closing token '{}' in list",
                    separator, end_token,
                ),
                next.unwrap(),
            );
        }
    }
    elements
}

fn parse_function_call<'a, I: TokIter<'a>>(
    ident_token_idx: usize,
    ident: &str,
    iter: &mut Peekable<I>,
) -> Option<AstNode> {
    let lparen_token = iter.peek();
    if !lparen_token.is_some_and(|t| t.kind == TokenKind::LParen) {
        return None;
    };
    iter.next();
    let args = parse_list(iter, TokenKind::Comma, TokenKind::RParen);
    Some(AstNode {
        token_idx: ident_token_idx,
        kind: AstNodeKind::FunctionCall(ident.into(), args),
    })
}

fn parse_block<'a, I: TokIter<'a>>(
    iter: &mut Peekable<I>,
    top_level: bool,
    is_loop: bool,
    extra_vars: Vec<String>,
) -> Box<AstNode> {
    let token_idx = if !top_level {
        let lbrace_token = iter.next().cloned();
        let Some(lbrace_token) = lbrace_token else {
            fatal("Expected '{' at start of block", iter.peek().unwrap());
        };
        if lbrace_token.kind != TokenKind::LBrace {
            fatal("Expected '{' at start of block", &lbrace_token);
        }
        lbrace_token.index
    } else {
        0
    };
    let mut block = Vec::new();
    let mut local_vars = Vec::new();
    for var in extra_vars {
        local_vars.push(var);
    }
    while let Some(token) = iter.peek().cloned() {
        if !top_level && token.kind == TokenKind::RBrace {
            iter.next();
            break;
        }
        if let Some(node) = parse_statement(iter, is_loop) {
            block.push(node);
            if let AstNodeKind::Assign(var_name, _) = &block.last().unwrap().kind
                && !local_vars.contains(var_name)
            {
                local_vars.push(var_name.to_string());
            }
        } else {
            fatal("Unexpected token in block", token);
        }
    }
    Box::new(AstNode {
        token_idx,
        kind: AstNodeKind::Block(block, local_vars),
    })
}

// If the iterable is an expression, it needs to be stored somwhere.
pub const FOR_ITERABLE_TEMP_VAR: &str = "__for_iterable_temp";

// Index needs to be stored somewhere.
pub const FOR_INDEX_TEMP_VAR: &str = "__for_index_temp";

// Even if not assigned to a variable, the key needs to be stored somewhere.
pub const FOR_KEY_TEMP_VAR: &str = "__for_key_temp";

fn parse_for_loop<'a, I: TokIter<'a>>(iter: &mut Peekable<I>) -> AstNode {
    let for_token = iter.next().unwrap();
    trace!("Parsing for loop, for token: {:?}", for_token);
    let token_idx = for_token.index;

    let mut token = iter.next().unwrap();
    trace!("Parsing for loop, first token: {:?}", token);
    let TokenKind::Ident(key_var) = &token.kind else {
        fatal("Expected identifier after 'for'", token);
    };
    let key_var = if key_var != "_" {
        Some(key_var.clone())
    } else {
        None
    };
    token = iter.next().unwrap();
    trace!("Parsing for loop, second token: {:?}", token);
    let item_var = if token.kind == TokenKind::Comma {
        token = iter.next().unwrap();
        trace!("Parsing for loop, item variable token: {:?}", token);
        let TokenKind::Ident(item_var) = &token.kind else {
            fatal("Expected identifier after comma in for loop", token);
        };
        token = iter.next().unwrap();
        if item_var != "_" {
            Some(item_var.clone())
        } else {
            None
        }
    } else {
        None
    };
    if token.kind != TokenKind::KeywordIn {
        fatal("Expected 'in' after item variable in for loop", token);
    }
    let collection_expr = parse_expression(iter).unwrap_or_else(|| {
        fatal(
            "Expected collection expression after 'in' in for loop",
            iter.peek().unwrap(),
        );
    });
    trace!(
        "Parsing for loop, collection expression: {:?}",
        collection_expr
    );

    let iterable_var = format!("{}{}", FOR_ITERABLE_TEMP_VAR, token_idx);
    let index_var = format!("{}{}", FOR_INDEX_TEMP_VAR, token_idx);
    let key_var = key_var.unwrap_or_else(|| format!("{}{}", FOR_KEY_TEMP_VAR, token_idx));

    let mut extra_vars = vec![iterable_var, index_var.clone(), key_var.clone()];
    if let Some(item_var) = &item_var {
        extra_vars.push(item_var.clone());
    }
    let body = parse_block(iter, false, true, extra_vars);

    AstNode {
        token_idx,
        kind: AstNodeKind::ForLoop(
            index_var,
            key_var,
            item_var,
            Box::new(collection_expr),
            body,
        ),
    }
}

fn parse_if_statement<'a, I: TokIter<'a>>(iter: &mut Peekable<I>, is_loop: bool) -> AstNode {
    let if_token = iter.next().unwrap();
    let token_idx = if_token.index;
    trace!("Parsing if expression, if token: {:?}", if_token);

    let condition = parse_expression(iter).unwrap_or_else(|| {
        fatal(
            "Expected condition expression after 'if'",
            iter.peek().unwrap(),
        );
    });
    let block = parse_block(iter, false, is_loop, vec![]);

    let else_token = iter.peek();
    let else_block = else_token
        .is_some_and(|t| t.kind == TokenKind::KeywordElse)
        .then(|| {
            iter.next();
            parse_block(iter, false, is_loop, vec![])
        });
    AstNode {
        token_idx,
        kind: AstNodeKind::IfStatement(Box::new(condition), block, else_block),
    }
}

fn parse_primary_expression<'a, I: TokIter<'a>>(iter: &mut Peekable<I>) -> Option<AstNode> {
    let token = iter.peek()?;
    let token_idx = token.index;

    match &token.kind {
        TokenKind::Operator(Operator::Sub) => {
            iter.next();
            let expr = parse_primary_expression(iter).unwrap_or_else(|| {
                fatal("Expected expression after '-'", iter.peek().unwrap());
            });
            Some(AstNode {
                token_idx,
                kind: AstNodeKind::BinaryOp(
                    Box::new(AstNode {
                        token_idx,
                        kind: AstNodeKind::Literal(Value::Integer(0)),
                    }),
                    Operator::Sub,
                    Box::new(expr),
                ),
            })
        }
        TokenKind::Literal(lit) => {
            iter.next();
            Some(AstNode {
                token_idx,
                kind: AstNodeKind::Literal(lit.clone()),
            })
        }
        TokenKind::Ident(ident) => {
            let ident_token_idx = iter.next().unwrap().index;
            trace!("Parsing identifier: {}", ident);
            if let Some(fcall) = parse_function_call(ident_token_idx, ident, iter) {
                Some(fcall)
            } else {
                Some(AstNode {
                    token_idx: ident_token_idx,
                    kind: AstNodeKind::Variable(ident.clone()),
                })
            }
        }
        TokenKind::LParen => {
            iter.next();
            let expr = parse_expression(iter).unwrap_or_else(|| {
                fatal("Expected expression after '('", iter.peek().unwrap());
            });
            let next = iter.peek().unwrap();
            if next.kind != TokenKind::RParen {
                fatal("Expected closing parenthesis", next);
            }
            iter.next();
            Some(expr)
        }
        _ => None,
    }
}

fn parse_expression<'a, I: TokIter<'a>>(iter: &mut Peekable<I>) -> Option<AstNode> {
    let left = parse_primary_expression(iter)?;
    trace!("Parsed primary expression: {:?}", left);
    parse_expression_impl(iter, left, 0)
}

fn parse_expression_impl<'a, I: TokIter<'a>>(
    iter: &mut Peekable<I>,
    mut left: AstNode,
    min_precedence: u8,
) -> Option<AstNode> {
    while let Some(tok) = iter.peek()
        && let TokenKind::Operator(op) = tok.kind
    {
        if op.precedence() < min_precedence {
            break;
        }
        let token_idx = tok.index;
        iter.next();
        let mut right = parse_primary_expression(iter).unwrap_or_else(|| {
            fatal("Expected expression after operator", iter.peek().unwrap());
        });

        trace!(
            "Parsed right-hand side expression: {:?} after op, {:?}",
            right, op
        );

        while let Some(next_tok) = iter.peek()
            && let TokenKind::Operator(next_op) = next_tok.kind
        {
            if next_op.precedence() > op.precedence() {
                right =
                    parse_expression_impl(iter, right, next_op.precedence()).unwrap_or_else(|| {
                        fatal("Expected expression after operator", iter.peek().unwrap());
                    });
                trace!(
                    "Updated right-hand side expression to: {:?} after parsing higher precedence op {:?}",
                    right, next_op
                );
            } else {
                break;
            }
        }

        left = AstNode {
            token_idx,
            kind: AstNodeKind::BinaryOp(Box::new(left), op, Box::new(right)),
        }
    }

    Some(left)
}

fn parse_statement<'a, I: TokIter<'a>>(iter: &mut Peekable<I>, is_loop: bool) -> Option<AstNode> {
    let token = iter.peek()?;

    let statement = match &token.kind {
        TokenKind::Ident(ident) => {
            let ident_token = iter.next();
            let ident_token_idx = ident_token.unwrap().index;
            match &iter.peek().unwrap().kind {
                TokenKind::Assign => {
                    trace!("Parsing assignment to {}", ident);
                    iter.next();
                    let expr = parse_expression(iter).unwrap();
                    AstNode {
                        token_idx: ident_token_idx,
                        kind: AstNodeKind::Assign(ident.clone(), Box::new(expr)),
                    }
                }
                _ => {
                    trace!("Parsing function call starting with identifier {}", ident);
                    parse_function_call(ident_token_idx, ident, iter).unwrap_or_else(|| {
                        fatal("Unexpected token after ident", iter.peek().unwrap());
                    })
                }
            }
        }
        TokenKind::KeywordFor => parse_for_loop(iter),
        TokenKind::KeywordIf => parse_if_statement(iter, is_loop),
        TokenKind::KeywordContinue if is_loop => {
            let continue_token = iter.next().unwrap();
            AstNode {
                token_idx: continue_token.index,
                kind: AstNodeKind::Continue,
            }
        }
        TokenKind::KeywordBreak if is_loop => {
            let break_token = iter.next().unwrap();
            AstNode {
                token_idx: break_token.index,
                kind: AstNodeKind::Break,
            }
        }
        _ => return None,
    };
    Some(statement)
}

fn fatal(msg: &str, token: &Token) -> ! {
    fatal_generic(msg, "Parsing terminated due to previous error.", token);
}

pub fn fatal_generic(msg: &str, end_msg: &str, token: &Token) -> ! {
    let char_col = find_source_char_col(token.line, token.byte_col);
    eprintln!(
        "Error: {}: at line {}, column {}",
        msg,
        token.line + 1,
        char_col + 1
    );
    report_source_pos(token.line, char_col);
    eprintln!("{}", end_msg);
    std::process::exit(1);
}

pub fn parse(tokens: &[Token]) -> Box<AstNode> {
    let mut iter = tokens.iter().peekable();

    let block = parse_block(&mut iter, true, false, vec![]);

    if iter.peek().is_some() {
        fatal(
            "Unexpected token after end of program",
            iter.peek().unwrap(),
        );
    }

    block
}
