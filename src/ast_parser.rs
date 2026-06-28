use std::cell::RefCell;
use std::iter::Peekable;

use bumpalo::Bump;
use colored::Colorize;
use log::trace;

use crate::tokenizer::{
    Literal, Operator, SourcePos, Token, TokenKind, find_source_char_col, report_source,
};

/// Bundles shared state needed throughout the parser: the bump allocator,
/// the source text, and the full token list (for error reporting).
pub struct ParseCtx<'b> {
    pub bump: &'b Bump,
    pub source: &'b str,
    pub tokens: &'b [Token<'b>],
    /// Shared scratch buffer for collecting AST nodes before freezing them into
    /// bump-allocated slices.
    scratch: RefCell<Vec<AstNode<'b>>>,
}

fn fatal_at_last_token(ctx: &ParseCtx, msg: &str) -> ! {
    let last_token = ctx.tokens.iter().rfind(|t| t.kind != TokenKind::Comment);
    if let Some(token) = last_token {
        fatal(ctx, msg, token);
    } else {
        eprintln!("{}: Empty file", "Error".color(colored::Color::BrightRed));
        eprintln!("Parsing terminated due to previous error.");
        std::panic::panic_any(crate::LangError("Empty file".to_string()));
    }
}

fn expect_next<'b, I: TokIter<'b>>(
    ctx: &ParseCtx,
    iter: &mut Peekable<I>,
    msg: &str,
) -> &'b Token<'b> {
    iter.next().unwrap_or_else(|| fatal_at_last_token(ctx, msg))
}

fn expect_next_token<'b, I: TokIter<'b>>(
    ctx: &ParseCtx,
    iter: &mut Peekable<I>,
    expected: TokenKind,
    msg: &str,
) -> &'b Token<'b> {
    let token = expect_next(ctx, iter, msg);
    if token.kind != expected {
        fatal(ctx, &format!("Expected token '{}'", expected), token);
    }
    token
}

fn expect_peek<'b, I: TokIter<'b>>(
    ctx: &ParseCtx,
    iter: &mut Peekable<I>,
    msg: &str,
) -> &'b Token<'b> {
    iter.peek()
        .copied()
        .unwrap_or_else(|| fatal_at_last_token(ctx, msg))
}

#[derive(Debug)]
pub enum AstNodeKind<'a> {
    DeclareAssign {
        name: &'a str,
        expr: &'a AstNode<'a>,
    },
    Assign {
        name: &'a str,
        expr: &'a AstNode<'a>,
    },
    /// name, arguments
    FunctionCall {
        name: &'a str,
        args: &'a [AstNode<'a>],
    },
    FunctionDefinition {
        name: &'a str,
        params: &'a [AstNode<'a>],
        body: &'a AstNode<'a>,
    },
    Return {
        expr: &'a AstNode<'a>,
    },
    ForLoop {
        key: Option<&'a AstNode<'a>>,
        item: Option<&'a AstNode<'a>>,
        iterable: &'a AstNode<'a>,
        body: &'a AstNode<'a>,
    },
    WhileLoop {
        condition: &'a AstNode<'a>,
        body: &'a AstNode<'a>,
    },
    Declaration(&'a str),
    Continue,
    Break,
    IfStatement {
        condition: &'a AstNode<'a>,
        body: &'a AstNode<'a>,
        else_body: Option<&'a AstNode<'a>>,
    },
    BinaryOp {
        left: &'a AstNode<'a>,
        op: Operator,
        right: &'a AstNode<'a>,
    },
    Block(&'a [AstNode<'a>]),
    Literal(Literal<'a>),
    Variable(&'a str),
    /// target[key] - subscript read
    Subscript {
        target: &'a AstNode<'a>,
        key: &'a AstNode<'a>,
    },
    /// target[key] = value - subscript write
    SubscriptAssign {
        target: &'a AstNode<'a>,
        key: &'a AstNode<'a>,
        value: &'a AstNode<'a>,
    },
}

#[derive(Debug)]
pub struct AstNode<'a> {
    pub token_idx: usize,
    pub kind: AstNodeKind<'a>,
}

impl<'a> AstNode<'a> {
    pub fn get_var_name(&self) -> Option<&'a str> {
        match self.kind {
            AstNodeKind::DeclareAssign { name, .. }
            | AstNodeKind::Assign { name, .. }
            | AstNodeKind::Variable(name)
            | AstNodeKind::Declaration(name) => Some(name),
            _ => None,
        }
    }
}

impl<'b> ParseCtx<'b> {
    /// Saves the current scratch buffer length. The caller should push items,
    /// then call `scratch_take_since` to drain and bump-allocate only their portion.
    fn scratch_start(&self) -> usize {
        self.scratch.borrow().len()
    }

    fn scratch_push(&self, node: AstNode<'b>) {
        self.scratch.borrow_mut().push(node);
    }

    /// Drains items pushed since `start`, bump-allocates them as a slice,
    /// and restores the scratch buffer to its previous length.
    fn scratch_take_since(&self, start: usize) -> &'b [AstNode<'b>] {
        let mut scratch = self.scratch.borrow_mut();
        self.bump.alloc_slice_fill_iter(scratch.drain(start..))
    }
}

trait TokIter<'a>: Iterator<Item = &'a Token<'a>> + Clone {}
impl<'a, T: Iterator<Item = &'a Token<'a>> + Clone> TokIter<'a> for T {}

fn parse_list<'b, I: TokIter<'b>>(
    ctx: &ParseCtx<'b>,
    iter: &mut Peekable<I>,
    separator: TokenKind,
    end_token: TokenKind,
    eof_msg: &str,
    mut collect_fn: impl FnMut(&'b Token<'b>, AstNode<'b>),
) {
    loop {
        let next = iter.peek();
        if next.is_some_and(|t| t.kind == end_token) {
            iter.next();
            break;
        }

        let tok = expect_peek(ctx, iter, eof_msg);
        let element = parse_expression(ctx, iter)
            .unwrap_or_else(|| fatal(ctx, "Expected expression in list", tok));
        collect_fn(tok, element);
        let next = iter.peek();
        if next.is_some_and(|t| t.kind == separator) {
            iter.next();
        } else if next.is_some_and(|t| t.kind == end_token) {
            iter.next();
            break;
        } else {
            let err_tok = expect_peek(ctx, iter, eof_msg);
            fatal(
                ctx,
                &format!(
                    "Expected separator '{}' or closing token '{}' in list",
                    separator, end_token,
                ),
                err_tok,
            );
        }
    }
}

fn parse_function_call<'b, I: TokIter<'b>>(
    ctx: &ParseCtx<'b>,
    ident_token_idx: usize,
    ident: &'b str,
    iter: &mut Peekable<I>,
) -> Option<AstNode<'b>> {
    let lparen_token = iter.peek();
    if !lparen_token.is_some_and(|t| t.kind == TokenKind::LParen) {
        return None;
    };
    iter.next();
    let scratch_start = ctx.scratch_start();
    parse_list(
        ctx,
        iter,
        TokenKind::Comma,
        TokenKind::RParen,
        "Expected ')' to close function call",
        |_, arg_node| {
            ctx.scratch_push(arg_node);
        },
    );
    Some(AstNode {
        token_idx: ident_token_idx,
        kind: AstNodeKind::FunctionCall {
            name: ident,
            args: ctx.scratch_take_since(scratch_start),
        },
    })
}

fn parse_block<'b, I: TokIter<'b>>(
    ctx: &ParseCtx<'b>,
    iter: &mut Peekable<I>,
    top_level: bool,
    in_loop: bool,
    in_function: bool,
) -> &'b AstNode<'b> {
    let token_idx = if !top_level {
        let lbrace_token = expect_next_token(
            ctx,
            iter,
            TokenKind::LBrace,
            "Expected '{' at start of block",
        );
        lbrace_token.index
    } else {
        0
    };
    let scratch_start = ctx.scratch_start();
    let mut found_rbrace = false;

    while let Some(token) = iter.peek().cloned() {
        if !top_level && token.kind == TokenKind::RBrace {
            iter.next();
            found_rbrace = true;
            break;
        }
        if let Some(node) = parse_statement(ctx, iter, top_level, in_loop, in_function) {
            ctx.scratch_push(node);
        } else {
            fatal(ctx, "Unexpected token in block", token);
        }
    }

    if !top_level && !found_rbrace {
        fatal_at_last_token(ctx, "Expected '}' to close block");
    }

    ctx.bump.alloc(AstNode {
        token_idx,
        kind: AstNodeKind::Block(ctx.scratch_take_since(scratch_start)),
    })
}

fn parse_function_definition<'b, I: TokIter<'b>>(
    ctx: &ParseCtx<'b>,
    iter: &mut Peekable<I>,
) -> AstNode<'b> {
    let fn_token = expect_next_token(ctx, iter, TokenKind::KeywordFn, "Expected 'fn' keyword");
    trace!("Parsing function definition, fn token: {:?}", fn_token);
    let token_idx = fn_token.index;

    let name_token = expect_next(ctx, iter, "Expected function name after 'fn'");
    let TokenKind::Ident(func_name) = name_token.kind else {
        fatal(ctx, "Expected function name after 'fn'", name_token);
    };
    trace!("Parsing function definition, name: {}", func_name);

    expect_next_token(
        ctx,
        iter,
        TokenKind::LParen,
        "Expected '(' after function name",
    );

    let scratch_start = ctx.scratch_start();
    parse_list(
        ctx,
        iter,
        TokenKind::Comma,
        TokenKind::RParen,
        "Expected ')' to close function parameter list",
        |tok, param_node| {
            let AstNodeKind::Variable(v) = param_node.kind else {
                fatal(
                    ctx,
                    "Expected parameter name in parens of function definition",
                    tok,
                );
            };
            ctx.scratch_push(AstNode {
                token_idx: param_node.token_idx,
                kind: AstNodeKind::Declaration(v),
            });
        },
    );

    let params = ctx.scratch_take_since(scratch_start);
    trace!("Parsing function definition, parameters: {:?}", params);

    let body = parse_block(ctx, iter, false, false, true);

    AstNode {
        token_idx,
        kind: AstNodeKind::FunctionDefinition {
            name: func_name,
            params,
            body,
        },
    }
}

fn parse_for_loop<'b, I: TokIter<'b>>(
    ctx: &ParseCtx<'b>,
    iter: &mut Peekable<I>,
    in_function: bool,
) -> AstNode<'b> {
    let for_token = expect_next_token(ctx, iter, TokenKind::KeywordFor, "Expected 'for' keyword");
    trace!("Parsing for loop, for token: {:?}", for_token);
    let token_idx = for_token.index;

    let key_token = expect_next(ctx, iter, "Expected loop variable after 'for'");
    trace!("Parsing for loop, first token: {:?}", key_token);
    let TokenKind::Ident(key_var) = key_token.kind else {
        fatal(ctx, "Expected identifier after 'for'", key_token);
    };
    let key = if key_var != "_" {
        Some(ctx.bump.alloc(AstNode {
            token_idx: key_token.index,
            kind: AstNodeKind::Declaration(key_var),
        }) as &_)
    } else {
        None
    };

    let mut next_token = expect_next(ctx, iter, "Expected 'in' or ',' after loop variable");
    trace!("Parsing for loop, second token: {:?}", next_token);
    let item = if next_token.kind == TokenKind::Comma {
        let item_token = expect_next(ctx, iter, "Expected identifier after ','");
        trace!("Parsing for loop, item variable token: {:?}", item_token);
        let TokenKind::Ident(item_var) = item_token.kind else {
            fatal(ctx, "Expected identifier after ','", item_token);
        };
        next_token = expect_next(ctx, iter, "Expected 'in' keyword");
        if item_var == "_" {
            None
        } else {
            Some(ctx.bump.alloc(AstNode {
                token_idx: item_token.index,
                kind: AstNodeKind::Declaration(item_var),
            }) as &_)
        }
    } else {
        None
    };

    if next_token.kind != TokenKind::KeywordIn {
        fatal(
            ctx,
            "Expected 'in' after item variable in for loop",
            next_token,
        );
    }
    let collection_expr = parse_expression(ctx, iter).unwrap_or_else(|| {
        let tok = expect_peek(ctx, iter, "Expected collection expression after 'in'");
        fatal(
            ctx,
            "Expected collection expression after 'in' in for loop",
            tok,
        );
    });
    trace!(
        "Parsing for loop, collection expression: {:?}",
        collection_expr
    );

    let body = parse_block(ctx, iter, false, true, in_function);

    AstNode {
        token_idx,
        kind: AstNodeKind::ForLoop {
            key,
            item,
            iterable: ctx.bump.alloc(collection_expr),
            body,
        },
    }
}

fn parse_while_loop<'b, I: TokIter<'b>>(
    ctx: &ParseCtx<'b>,
    iter: &mut Peekable<I>,
    in_function: bool,
) -> AstNode<'b> {
    let while_token = expect_next_token(
        ctx,
        iter,
        TokenKind::KeywordWhile,
        "Expected 'while' keyword",
    );
    trace!("Parsing while loop, while token: {:?}", while_token);
    let token_idx = while_token.index;

    let expr = parse_expression(ctx, iter).unwrap_or_else(|| {
        let tok = expect_peek(ctx, iter, "Expected condition expression after 'while'");
        fatal(ctx, "Expected condition expression after 'while'", tok);
    });

    let body = parse_block(ctx, iter, false, true, in_function);

    AstNode {
        token_idx,
        kind: AstNodeKind::WhileLoop {
            condition: ctx.bump.alloc(expr),
            body,
        },
    }
}

fn parse_if_statement<'b, I: TokIter<'b>>(
    ctx: &ParseCtx<'b>,
    iter: &mut Peekable<I>,
    in_loop: bool,
    in_function: bool,
) -> AstNode<'b> {
    let if_token = expect_next_token(ctx, iter, TokenKind::KeywordIf, "Expected 'if' keyword");
    let token_idx = if_token.index;
    trace!("Parsing if expression, if token: {:?}", if_token);

    let condition = parse_expression(ctx, iter).unwrap_or_else(|| {
        let tok = expect_peek(ctx, iter, "Expected condition expression after 'if'");
        fatal(ctx, "Expected condition expression after 'if'", tok);
    });
    let body = parse_block(ctx, iter, false, in_loop, in_function);

    let else_token = iter.peek();
    let else_body = else_token
        .is_some_and(|t| t.kind == TokenKind::KeywordElse)
        .then(|| {
            iter.next();
            parse_block(ctx, iter, false, in_loop, in_function)
        });
    AstNode {
        token_idx,
        kind: AstNodeKind::IfStatement {
            condition: ctx.bump.alloc(condition),
            body,
            else_body,
        },
    }
}

fn parse_primary_expression<'b, I: TokIter<'b>>(
    ctx: &ParseCtx<'b>,
    iter: &mut Peekable<I>,
) -> Option<AstNode<'b>> {
    let token = iter.peek()?;
    let token_idx = token.index;

    let mut atom = match &token.kind {
        TokenKind::Operator(Operator::Sub) => {
            iter.next();
            let expr = parse_primary_expression(ctx, iter).unwrap_or_else(|| {
                let tok = expect_peek(ctx, iter, "Expected expression after '-'");
                fatal(ctx, "Expected expression after '-'", tok);
            });
            AstNode {
                token_idx,
                kind: AstNodeKind::BinaryOp {
                    left: ctx.bump.alloc(AstNode {
                        token_idx,
                        kind: AstNodeKind::Literal(Literal::Number(0.)),
                    }),
                    op: Operator::Sub,
                    right: ctx.bump.alloc(expr),
                },
            }
        }
        TokenKind::Literal(lit) => {
            iter.next();
            AstNode {
                token_idx,
                kind: AstNodeKind::Literal(lit.clone()),
            }
        }
        TokenKind::Ident(ident) => {
            let ident = *ident;
            let ident_token_idx = iter.next().unwrap().index;
            trace!("Parsing identifier: {}", ident);
            if let Some(fcall) = parse_function_call(ctx, ident_token_idx, ident, iter) {
                fcall
            } else {
                AstNode {
                    token_idx: ident_token_idx,
                    kind: AstNodeKind::Variable(ident),
                }
            }
        }
        TokenKind::LParen => {
            iter.next();
            let expr = parse_expression(ctx, iter).unwrap_or_else(|| {
                let tok = expect_peek(ctx, iter, "Expected expression after '('");
                fatal(ctx, "Expected expression after '('", tok);
            });
            let next = expect_peek(ctx, iter, "Expected ')' to close parenthesized expression");
            if next.kind != TokenKind::RParen {
                fatal(ctx, "Expected closing parenthesis", next);
            }
            iter.next();
            expr
        }
        _ => None?,
    };

    while let Some(TokenKind::LBracket) = iter.peek().map(|t| &t.kind) {
        let bracket_token_idx = iter.next().unwrap().index;
        let key_expr = parse_expression(ctx, iter).unwrap_or_else(|| {
            let tok = expect_peek(ctx, iter, "Expected expression after '['");
            fatal(ctx, "Expected expression after '['", tok);
        });
        let next = expect_peek(ctx, iter, "Expected ']' to close subscript");
        if next.kind != TokenKind::RBracket {
            fatal(ctx, "Expected ']' after expression", next);
        }
        iter.next();
        atom = AstNode {
            token_idx: bracket_token_idx,
            kind: AstNodeKind::Subscript {
                target: ctx.bump.alloc(atom),
                key: ctx.bump.alloc(key_expr),
            },
        };
    }

    Some(atom)
}

fn parse_expression<'b, I: TokIter<'b>>(
    ctx: &ParseCtx<'b>,
    iter: &mut Peekable<I>,
) -> Option<AstNode<'b>> {
    let left = parse_primary_expression(ctx, iter)?;
    trace!("Parsed primary expression: {:?}", left);
    parse_expression_impl(ctx, iter, left, 0)
}

fn parse_expression_impl<'b, I: TokIter<'b>>(
    ctx: &ParseCtx<'b>,
    iter: &mut Peekable<I>,
    mut left: AstNode<'b>,
    min_precedence: u8,
) -> Option<AstNode<'b>> {
    while let Some(tok) = iter.peek()
        && let TokenKind::Operator(op) = tok.kind
    {
        if op.precedence() < min_precedence {
            break;
        }
        let token_idx = tok.index;
        iter.next();
        let mut right = parse_primary_expression(ctx, iter).unwrap_or_else(|| {
            let err_tok = expect_peek(ctx, iter, "Expected expression after operator");
            fatal(ctx, "Expected expression after operator", err_tok);
        });

        trace!(
            "Parsed right-hand side expression: {:?} after op, {:?}",
            right, op
        );

        while let Some(next_tok) = iter.peek()
            && let TokenKind::Operator(next_op) = next_tok.kind
        {
            if next_op.precedence() > op.precedence() {
                right = parse_expression_impl(ctx, iter, right, next_op.precedence())
                    .unwrap_or_else(|| {
                        let err_tok = expect_peek(ctx, iter, "Expected expression after operator");
                        fatal(ctx, "Expected expression after operator", err_tok);
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
            kind: AstNodeKind::BinaryOp {
                left: ctx.bump.alloc(left),
                op,
                right: ctx.bump.alloc(right),
            },
        }
    }

    Some(left)
}

fn parse_statement<'b, I: TokIter<'b>>(
    ctx: &ParseCtx<'b>,
    iter: &mut Peekable<I>,
    top_level: bool,
    in_loop: bool,
    in_function: bool,
) -> Option<AstNode<'b>> {
    let token = iter.peek()?;

    let statement = match &token.kind {
        TokenKind::Ident(ident) => {
            let ident = *ident;
            let ident_token = iter.next();
            let ident_token_idx = ident_token.unwrap().index;

            match iter.peek().map(|t| &t.kind) {
                Some(TokenKind::DeclareAssign) => {
                    trace!("Parsing declaration and assignment of {}", ident);
                    iter.next();
                    let expr = parse_expression(ctx, iter).unwrap_or_else(|| {
                        let tok = expect_peek(ctx, iter, "Expected expression after ':='");
                        fatal(ctx, "Expected expression after ':='", tok);
                    });
                    AstNode {
                        token_idx: ident_token_idx,
                        kind: AstNodeKind::DeclareAssign {
                            name: ident,
                            expr: ctx.bump.alloc(expr),
                        },
                    }
                }
                Some(TokenKind::Assign) => {
                    trace!("Parsing assignment to {}", ident);
                    iter.next();
                    let expr = parse_expression(ctx, iter).unwrap_or_else(|| {
                        let tok = expect_peek(ctx, iter, "Expected expression after '='");
                        fatal(ctx, "Expected expression after '='", tok);
                    });
                    AstNode {
                        token_idx: ident_token_idx,
                        kind: AstNodeKind::Assign {
                            name: ident,
                            expr: ctx.bump.alloc(expr),
                        },
                    }
                }
                Some(TokenKind::LBracket) => {
                    trace!(
                        "Parsing potential subscript assignment starting with {}",
                        ident
                    );

                    let mut target = AstNode {
                        token_idx: ident_token_idx,
                        kind: AstNodeKind::Variable(ident),
                    };

                    let mut last_bracket_idx = ident_token_idx;
                    let mut last_key: Option<AstNode<'b>> = None;

                    while iter.peek().is_some_and(|t| t.kind == TokenKind::LBracket) {
                        let bracket_token_idx = iter.next().unwrap().index;
                        let key_expr = parse_expression(ctx, iter).unwrap_or_else(|| {
                            let tok = expect_peek(ctx, iter, "Expected expression after '['");
                            fatal(ctx, "Expected expression after '['", tok);
                        });
                        let next = expect_peek(ctx, iter, "Expected ']' to close subscript");
                        if next.kind != TokenKind::RBracket {
                            fatal(ctx, "Expected ']' after expression", next);
                        }
                        iter.next();

                        if let Some(prev_key) = last_key.take() {
                            target = AstNode {
                                token_idx: last_bracket_idx,
                                kind: AstNodeKind::Subscript {
                                    target: ctx.bump.alloc(target),
                                    key: ctx.bump.alloc(prev_key),
                                },
                            };
                        }

                        last_bracket_idx = bracket_token_idx;
                        last_key = Some(key_expr);
                    }

                    if iter.peek().is_some_and(|t| t.kind == TokenKind::Assign) {
                        iter.next();
                        let value_expr = parse_expression(ctx, iter).unwrap_or_else(|| {
                            let tok = expect_peek(ctx, iter, "Expected expression after '='");
                            fatal(ctx, "Expected expression after '='", tok);
                        });
                        AstNode {
                            token_idx: last_bracket_idx,
                            kind: AstNodeKind::SubscriptAssign {
                                target: ctx.bump.alloc(target),
                                key: ctx.bump.alloc(last_key.unwrap()),
                                value: ctx.bump.alloc(value_expr),
                            },
                        }
                    } else {
                        let tok = expect_peek(ctx, iter, "Expected '=' after subscript expression");
                        fatal(
                            ctx,
                            "Expected '=' after subscript expression in statement",
                            tok,
                        );
                    }
                }
                _ => {
                    trace!("Parsing function call starting with identifier {}", ident);
                    parse_function_call(ctx, ident_token_idx, ident, iter).unwrap_or_else(|| {
                        let tok = expect_peek(
                            ctx,
                            iter,
                            "Expected '(' for function call or operator after identifier",
                        );
                        fatal(ctx, "Unexpected token after ident", tok);
                    })
                }
            }
        }
        TokenKind::KeywordFn if top_level => parse_function_definition(ctx, iter),
        TokenKind::KeywordReturn if in_function => {
            let return_token = expect_next(ctx, iter, "Expected 'return' keyword");
            let expr = parse_expression(ctx, iter).unwrap_or_else(|| {
                let tok = expect_peek(ctx, iter, "Expected expression after 'return'");
                fatal(ctx, "Expected expression after 'return'", tok);
            });
            AstNode {
                token_idx: return_token.index,
                kind: AstNodeKind::Return {
                    expr: ctx.bump.alloc(expr),
                },
            }
        }
        TokenKind::KeywordFor => parse_for_loop(ctx, iter, in_function),
        TokenKind::KeywordWhile => parse_while_loop(ctx, iter, in_function),
        TokenKind::KeywordIf => parse_if_statement(ctx, iter, in_loop, in_function),
        TokenKind::KeywordContinue if in_loop => {
            let continue_token = expect_next(ctx, iter, "Expected 'continue' keyword");
            AstNode {
                token_idx: continue_token.index,
                kind: AstNodeKind::Continue,
            }
        }
        TokenKind::KeywordBreak if in_loop => {
            let break_token = expect_next(ctx, iter, "Expected 'break' keyword");
            AstNode {
                token_idx: break_token.index,
                kind: AstNodeKind::Break,
            }
        }
        _ => return None,
    };
    Some(statement)
}

fn fatal(ctx: &ParseCtx, msg: &str, token: &Token<'_>) -> ! {
    fatal_generic(
        ctx.source,
        ctx.tokens,
        msg,
        "Parsing terminated due to previous error.",
        token,
    );
}

pub fn fatal_generic(
    source: &str,
    tokens: &[Token<'_>],
    msg: &str,
    end_msg: &str,
    token: &Token<'_>,
) -> ! {
    fatal_with_stack(source, tokens, msg, end_msg, token, &[]);
}

/// Represents a function call location for error reporting.
pub struct CallLocation<'a> {
    pub function_name: &'a str,
    pub line: usize,
}

/// `call_stack` should be in order from innermost to outermost.
pub fn fatal_with_stack(
    source: &str,
    tokens: &[Token<'_>],
    msg: &str,
    end_msg: &str,
    token: &Token<'_>,
    call_stack: &[CallLocation],
) -> ! {
    let char_col = find_source_char_col(source, token.line, token.byte_col);

    eprintln!(
        "{}: {}: at line {}, column {}",
        "Error".color(colored::Color::BrightRed),
        msg,
        token.line + 1,
        char_col + 1,
    );
    report_source(
        source,
        tokens,
        Some((
            SourcePos {
                row: token.line,
                char_col,
                byte_pos_start: token.byte_pos_start,
                byte_pos_end: token.byte_pos_end,
            },
            2,
            colored::Color::BrightRed,
        )),
    );

    if !call_stack.is_empty() {
        eprintln!();
        eprintln!("Call stack:");

        const MAX_TOP: usize = 10;
        const MAX_BOTTOM: usize = 5;

        let total = call_stack.len();
        let show_all = total <= MAX_TOP + MAX_BOTTOM;

        for (i, loc) in call_stack.iter().enumerate() {
            if show_all || i < MAX_TOP || i >= total - MAX_BOTTOM {
                eprintln!(
                    "  {}: {} at line {}",
                    i,
                    loc.function_name.color(TokenKind::Ident("").color()),
                    loc.line + 1,
                );
            } else if i == MAX_TOP {
                eprintln!("  ... {} more ...", total - MAX_TOP - MAX_BOTTOM);
            }
        }
    }

    eprintln!("{}", end_msg);
    std::panic::panic_any(crate::LangError(format!(
        "{}: at line {}, column {}",
        msg,
        token.line + 1,
        char_col + 1
    )));
}

pub fn parse<'b>(bump: &'b Bump, source: &'b str, tokens: &'b [Token<'b>]) -> &'b AstNode<'b> {
    let ctx = ParseCtx {
        bump,
        source,
        tokens,
        scratch: RefCell::new(Vec::new()),
    };

    let mut iter = tokens
        .iter()
        .filter(|t| t.kind != TokenKind::Comment)
        .peekable();

    let block = parse_block(&ctx, &mut iter, true, false, false);

    if iter.peek().is_some() {
        fatal(
            &ctx,
            "Unexpected token after end of program",
            iter.peek().unwrap(),
        );
    }

    block
}
