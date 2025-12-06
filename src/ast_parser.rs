use std::iter::Peekable;

use log::trace;

use crate::tokenizer::{Operator, Token, TokenKind, Value};

#[derive(Debug)]
pub enum AstNode {
    Assign(String, Box<AstNode>),
    FunctionCall(String, Vec<AstNode>),
    /// index, item, list, body
    ForLoop(String, Option<String>, Box<AstNode>, Vec<AstNode>),
    IfExpression(Box<AstNode>, Vec<AstNode>, Vec<AstNode>),
    BinaryOp(Box<AstNode>, Operator, Box<AstNode>),
    List(Vec<AstNode>),
    Literal(Value),
    Variable(String),
}

trait TokIter<'a>: Iterator<Item = &'a Token<'a>> + Clone {}
impl<'a, T: Iterator<Item = &'a Token<'a>> + Clone> TokIter<'a> for T {}

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
        let element = parse_expression(iter).unwrap();
        elements.push(element);
        let next = iter.peek();
        if next.is_some_and(|t| t.kind == separator) {
            iter.next();
        } else if next.is_some_and(|t| t.kind == end_token) {
            iter.next();
            break;
        } else {
            panic!("Expected ',' or ']/)' in list");
        }
    }
    elements
}

fn parse_function_call<'a, I: TokIter<'a>>(ident: &str, iter: &mut Peekable<I>) -> Option<AstNode> {
    let next = iter.peek();
    if !next.is_some_and(|t| t.kind == TokenKind::LParen) {
        return None;
    };
    iter.next();
    let args = parse_list(iter, TokenKind::Comma, TokenKind::RParen);
    Some(AstNode::FunctionCall(ident.to_owned(), args))
}

fn parse_block<'a, I: TokIter<'a>>(iter: &mut Peekable<I>) -> Vec<AstNode> {
    if !iter.next().is_some_and(|t| t.kind == TokenKind::LBrace) {
        panic!("Expected '{{' at start of block");
    }
    let mut block = Vec::new();
    while let Some(token) = iter.peek().cloned() {
        if token.kind == TokenKind::RBrace {
            iter.next();
            break;
        }
        let stmt = parse_statement(iter);
        if let Some(node) = stmt {
            block.push(node);
        } else {
            panic!("Unexpected token in block: {:?}", token);
        }
    }
    block
}

fn parse_for_loop<'a, I: TokIter<'a>>(iter: &mut Peekable<I>) -> AstNode {
    let mut token = iter.next().unwrap();
    trace!("Parsing for loop, first token: {:?}", token);
    let TokenKind::Ident(index_var) = &token.kind else {
        panic!("Expected identifier after 'for', found: {:?}", token);
    };
    token = iter.next().unwrap();
    trace!("Parsing for loop, second token: {:?}", token);
    let item_var = if token.kind == TokenKind::Comma {
        token = iter.next().unwrap();
        trace!("Parsing for loop, item variable token: {:?}", token);
        let TokenKind::Ident(item_var) = &token.kind else {
            panic!(
                "Expected identifier after for comma variable, found: {:?}",
                token
            );
        };
        token = iter.next().unwrap();
        Some(item_var.clone())
    } else {
        None
    };
    if token.kind != TokenKind::KeywordIn {
        panic!("Expected 'in' after item variable, found: {:?}", token);
    }
    let collection_expr = parse_expression(iter)
        .expect("Failed to parse collection expression after 'in' in for loop");

    let body = parse_block(iter);

    AstNode::ForLoop(
        index_var.clone(),
        item_var.clone(),
        Box::new(collection_expr),
        body,
    )
}

fn parse_if_expression<'a, I: TokIter<'a>>(iter: &mut Peekable<I>) -> AstNode {
    let condition = parse_expression(iter).expect("Expected a condition expression after 'if'");
    let block = parse_block(iter);
    let token = iter.peek();
    let mut else_block = Vec::new();
    if token.is_some_and(|t| t.kind == TokenKind::KeywordElse) {
        iter.next();
        else_block = parse_block(iter);
    }
    AstNode::IfExpression(Box::new(condition), block, else_block)
}

fn parse_primary_expression<'a, I: TokIter<'a>>(iter: &mut Peekable<I>) -> Option<AstNode> {
    let token = iter.peek()?;

    match &token.kind {
        TokenKind::Operator(Operator::Minus) => {
            iter.next();
            let expr =
                parse_primary_expression(iter).expect("Expected expression after unary minus");
            Some(AstNode::BinaryOp(
                Box::new(AstNode::Literal(Value::Integer(0))),
                Operator::Minus,
                Box::new(expr),
            ))
        }
        TokenKind::Literal(lit) => {
            iter.next();
            Some(AstNode::Literal(lit.clone()))
        }
        TokenKind::Ident(ident) => {
            iter.next();
            trace!("Parsing identifier: {}", ident);
            if let Some(fcall) = parse_function_call(ident, iter) {
                Some(fcall)
            } else {
                Some(AstNode::Variable(ident.clone()))
            }
        }
        TokenKind::LParen => {
            iter.next();
            let expr = parse_expression(iter).unwrap();
            let next = iter.peek().unwrap();
            if next.kind != TokenKind::RParen {
                panic!("Expected closing parenthesis, found: {:?}", next);
            }
            iter.next();
            Some(expr)
        }
        TokenKind::LSquareParen => {
            iter.next();
            let elements = parse_list(iter, TokenKind::Comma, TokenKind::RSquareParen);
            Some(AstNode::List(elements))
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
        iter.next();
        let mut right =
            parse_primary_expression(iter).expect("Expected an expression after operator");

        trace!(
            "Parsed right-hand side expression: {:?} after op, {:?}",
            right, op
        );

        while let Some(next_tok) = iter.peek()
            && let TokenKind::Operator(next_op) = next_tok.kind
        {
            if next_op.precedence() > op.precedence() {
                right = parse_expression_impl(iter, right, next_op.precedence())
                    .expect("Expected expression after operator");
                trace!(
                    "Updated right-hand side expression to: {:?} after parsing higher precedence op {:?}",
                    right, next_op
                );
            } else {
                break;
            }
        }

        left = AstNode::BinaryOp(Box::new(left), op, Box::new(right));
    }

    Some(left)
}

fn parse_statement<'a, I: TokIter<'a>>(iter: &mut Peekable<I>) -> Option<AstNode> {
    let token = iter.peek()?;

    let statement = match &token.kind {
        TokenKind::Ident(ident) => {
            iter.next();
            match &iter.peek().unwrap().kind {
                TokenKind::Assign => {
                    trace!("Parsing assignment to {}", ident);
                    iter.next();
                    let expr = parse_expression(iter).unwrap();
                    AstNode::Assign(ident.clone(), Box::new(expr))
                }
                t => {
                    trace!("Parsing function call starting with identifier {}", ident);
                    parse_function_call(ident, iter).unwrap_or_else(|| {
                        panic!("Unexpected token after identifier '{}': {:?}", ident, t)
                    })
                }
            }
        }
        TokenKind::KeywordFor => {
            iter.next();
            parse_for_loop(iter)
        }
        TokenKind::KeywordIf => {
            iter.next();
            parse_if_expression(iter)
        }
        _ => return None,
    };
    Some(statement)
}

pub fn parse(tokens: &[Token]) -> Vec<AstNode> {
    let mut iter = tokens.iter().peekable();

    let mut block = Vec::new();

    loop {
        let stmt = parse_statement(&mut iter);
        if let Some(node) = stmt {
            block.push(node);
        } else {
            break;
        }
    }

    if iter.peek().is_some() {
        panic!(
            "Unexpected tokens remaining after parsing: {:?}",
            iter.collect::<Vec<_>>()
        );
    }

    block
}
