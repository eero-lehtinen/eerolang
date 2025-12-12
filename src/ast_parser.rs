use std::iter::Peekable;

use log::trace;

use crate::{
    TOKENS,
    tokenizer::{Literal, Operator, Token, TokenKind, find_source_char_col, report_source_pos},
};

type VarName = String;

#[derive(Debug)]
pub enum AstNodeKind {
    DeclareAssign(VarName, Box<AstNode>),
    Assign(VarName, Box<AstNode>),
    /// name, arguments
    FunctionCall(VarName, Vec<AstNode>),
    /// name, parameters, body
    FunctionDefinition(VarName, Vec<AstNode>, Box<AstNode>),
    Return(Box<AstNode>),
    /// index, key, item, list, block
    ForLoop(
        Option<Box<AstNode>>,
        Option<Box<AstNode>>,
        Box<AstNode>,
        Box<AstNode>,
    ),
    /// condition, block
    WhileLoop(Box<AstNode>, Box<AstNode>),
    /// For loop and function parameter declarations.
    Declaration(VarName),
    Continue,
    Break,
    /// condition, then block, else block
    IfStatement(Box<AstNode>, Box<AstNode>, Option<Box<AstNode>>),
    BinaryOp(Box<AstNode>, Operator, Box<AstNode>),
    Block(Vec<AstNode>),
    Literal(Literal),
    Variable(VarName),
}

#[derive(Debug)]
pub struct AstNode {
    pub token_idx: usize,
    pub kind: AstNodeKind,
}

impl AstNode {
    pub fn get_var_name(&self) -> Option<&str> {
        match &self.kind {
            AstNodeKind::DeclareAssign(name, _)
            | AstNodeKind::Assign(name, _)
            | AstNodeKind::Variable(name)
            | AstNodeKind::Declaration(name) => Some(name),
            _ => None,
        }
    }
}

trait TokIter<'a>: Iterator<Item = &'a Token> + Clone {}
impl<'a, T: Iterator<Item = &'a Token> + Clone> TokIter<'a> for T {}

fn parse_list<'a, I: TokIter<'a>>(
    iter: &mut Peekable<I>,
    separator: TokenKind,
    end_token: TokenKind,
    mut collect_fn: impl FnMut(&'a Token, AstNode),
) {
    loop {
        let next = iter.peek();
        if next.is_some_and(|t| t.kind == end_token) {
            iter.next();
            break;
        }

        let tok = iter.peek().cloned();
        let element = parse_expression(iter)
            .unwrap_or_else(|| fatal("Expected expression in list", tok.unwrap()));
        collect_fn(tok.unwrap(), element);
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
    let mut args = Vec::new();
    parse_list(iter, TokenKind::Comma, TokenKind::RParen, |_, arg_node| {
        args.push(arg_node);
    });
    Some(AstNode {
        token_idx: ident_token_idx,
        kind: AstNodeKind::FunctionCall(ident.into(), args),
    })
}

fn parse_block<'a, I: TokIter<'a>>(
    iter: &mut Peekable<I>,
    top_level: bool,
    in_loop: bool,
    in_function: bool,
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

    while let Some(token) = iter.peek().cloned() {
        if !top_level && token.kind == TokenKind::RBrace {
            iter.next();
            break;
        }
        if let Some(node) = parse_statement(iter, top_level, in_loop, in_function) {
            block.push(node);
        } else {
            fatal("Unexpected token in block", token);
        }
    }
    Box::new(AstNode {
        token_idx,
        kind: AstNodeKind::Block(block),
    })
}

fn parse_function_definition<'a, I: TokIter<'a>>(iter: &mut Peekable<I>) -> AstNode {
    let fn_token = iter.next().unwrap();
    trace!("Parsing function definition, fn token: {:?}", fn_token);
    let token_idx = fn_token.index;

    let name_token = iter.next().unwrap();
    let TokenKind::Ident(func_name) = &name_token.kind else {
        fatal("Expected function name after 'fn'", name_token);
    };
    trace!("Parsing function definition, name: {}", func_name);

    let lparen_token = iter.next().unwrap();
    if lparen_token.kind != TokenKind::LParen {
        fatal("Expected '(' after function name", lparen_token);
    }

    let mut params = Vec::new();
    parse_list(
        iter,
        TokenKind::Comma,
        TokenKind::RParen,
        |tok, param_node| {
            let AstNodeKind::Variable(v) = param_node.kind else {
                fatal(
                    "Expected parameter name in parens of function definition",
                    tok,
                );
            };
            params.push(AstNode {
                token_idx: param_node.token_idx,
                kind: AstNodeKind::Declaration(v),
            });
        },
    );

    trace!("Parsing function definition, parameters: {:?}", params);

    let body = parse_block(iter, false, false, true);

    AstNode {
        token_idx,
        kind: AstNodeKind::FunctionDefinition(func_name.clone(), params, body),
    }
}

fn parse_for_loop<'a, I: TokIter<'a>>(iter: &mut Peekable<I>, in_function: bool) -> AstNode {
    let for_token = iter.next().unwrap();
    trace!("Parsing for loop, for token: {:?}", for_token);
    let token_idx = for_token.index;

    let key_token = iter.next().unwrap().clone();
    trace!("Parsing for loop, first token: {:?}", key_token);
    let TokenKind::Ident(key_var) = key_token.kind.clone() else {
        fatal("Expected identifier after 'for'", &key_token);
    };
    let key = if key_var != "_" {
        Some(Box::new(AstNode {
            token_idx: key_token.index,
            kind: AstNodeKind::Declaration(key_var),
        }))
    } else {
        None
    };

    let mut next_token = iter.next().unwrap();
    trace!("Parsing for loop, second token: {:?}", next_token);
    let item = if next_token.kind == TokenKind::Comma {
        let item_token = iter.next().unwrap();
        trace!("Parsing for loop, item variable token: {:?}", item_token);
        let TokenKind::Ident(item_var) = &item_token.kind else {
            fatal("Expected identifier after comma in for loop", item_token);
        };
        next_token = iter.next().unwrap();
        if item_var == "_" {
            None
        } else {
            Some(Box::new(AstNode {
                token_idx: item_token.index,
                kind: AstNodeKind::Declaration(item_var.clone()),
            }))
        }
    } else {
        None
    };

    if next_token.kind != TokenKind::KeywordIn {
        fatal("Expected 'in' after item variable in for loop", next_token);
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

    let body = parse_block(iter, false, true, in_function);

    AstNode {
        token_idx,
        kind: AstNodeKind::ForLoop(key, item, Box::new(collection_expr), body),
    }
}

fn parse_while_loop<'a, I: TokIter<'a>>(iter: &mut Peekable<I>, in_function: bool) -> AstNode {
    let while_token = iter.next().unwrap();
    trace!("Parsing while loop, while token: {:?}", while_token);
    let token_idx = while_token.index;

    let expr = parse_expression(iter).unwrap_or_else(|| {
        fatal(
            "Expected condition expression after 'while'",
            iter.peek().unwrap(),
        );
    });

    let body = parse_block(iter, false, true, in_function);

    AstNode {
        token_idx,
        kind: AstNodeKind::WhileLoop(Box::new(expr), body),
    }
}

fn parse_if_statement<'a, I: TokIter<'a>>(
    iter: &mut Peekable<I>,
    in_loop: bool,
    in_function: bool,
) -> AstNode {
    let if_token = iter.next().unwrap();
    let token_idx = if_token.index;
    trace!("Parsing if expression, if token: {:?}", if_token);

    let condition = parse_expression(iter).unwrap_or_else(|| {
        fatal(
            "Expected condition expression after 'if'",
            iter.peek().unwrap(),
        );
    });
    let block = parse_block(iter, false, in_loop, in_function);

    let else_token = iter.peek();
    let else_block = else_token
        .is_some_and(|t| t.kind == TokenKind::KeywordElse)
        .then(|| {
            iter.next();
            parse_block(iter, false, in_loop, in_function)
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
                        kind: AstNodeKind::Literal(Literal::Number(0.)),
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

fn parse_statement<'a, I: TokIter<'a>>(
    iter: &mut Peekable<I>,
    top_level: bool,
    in_loop: bool,
    in_function: bool,
) -> Option<AstNode> {
    let token = iter.peek()?;

    let statement = match &token.kind {
        TokenKind::Ident(ident) => {
            let ident_token = iter.next();
            let ident_token_idx = ident_token.unwrap().index;
            match &iter.peek().unwrap().kind {
                TokenKind::DeclareAssign => {
                    trace!("Parsing declaration and assignment of {}", ident);
                    iter.next();
                    let expr = parse_expression(iter).unwrap();
                    AstNode {
                        token_idx: ident_token_idx,
                        kind: AstNodeKind::DeclareAssign(ident.clone(), Box::new(expr)),
                    }
                }
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
        TokenKind::KeywordFn if top_level => parse_function_definition(iter),
        TokenKind::KeywordReturn if in_function => {
            let return_token = iter.next().unwrap();
            let expr = parse_expression(iter).unwrap_or_else(|| {
                fatal("Expected expression after 'return'", iter.peek().unwrap());
            });
            AstNode {
                token_idx: return_token.index,
                kind: AstNodeKind::Return(Box::new(expr)),
            }
        }
        TokenKind::KeywordFor => parse_for_loop(iter, in_function),
        TokenKind::KeywordWhile => parse_while_loop(iter, in_function),
        TokenKind::KeywordIf => parse_if_statement(iter, in_loop, in_function),
        TokenKind::KeywordContinue if in_loop => {
            let continue_token = iter.next().unwrap();
            AstNode {
                token_idx: continue_token.index,
                kind: AstNodeKind::Continue,
            }
        }
        TokenKind::KeywordBreak if in_loop => {
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
        char_col + 1,
    );
    report_source_pos(
        TOKENS.get().unwrap(),
        token.line,
        char_col,
        token.byte_pos_start,
        token.byte_pos_end,
        2,
    );
    eprintln!("{}", end_msg);
    std::process::exit(1);
}

pub fn parse(tokens: &[Token]) -> Box<AstNode> {
    let mut iter = tokens
        .iter()
        .filter(|t| t.kind != TokenKind::Comment)
        .peekable();

    let block = parse_block(&mut iter, true, false, false);

    if iter.peek().is_some() {
        fatal(
            "Unexpected token after end of program",
            iter.peek().unwrap(),
        );
    }

    block
}
