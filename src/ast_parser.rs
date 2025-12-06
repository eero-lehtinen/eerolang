use std::iter::Peekable;

use log::trace;

use crate::tokenizer::{Operator, Token, Value};

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

trait TokIter<'a>: Iterator<Item = &'a Token> + Clone {}
impl<'a, T: Iterator<Item = &'a Token> + Clone> TokIter<'a> for T {}

fn parse_list<'a, I: TokIter<'a>>(
    iter: &mut Peekable<I>,
    separator: Token,
    end_token: Token,
) -> Vec<AstNode> {
    let mut elements = Vec::new();
    loop {
        let next = iter.peek();
        if next == Some(&&end_token) {
            iter.next();
            break;
        }
        let element = parse_expression(iter).unwrap();
        elements.push(element);
        let next = iter.peek();
        if next == Some(&&separator) {
            iter.next();
        } else if next == Some(&&end_token) {
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
    let Some(Token::LParen) = next else {
        return None;
    };
    iter.next();
    let args = parse_list(iter, Token::Comma, Token::RParen);
    Some(AstNode::FunctionCall(ident.to_owned(), args))
}

fn parse_block<'a, I: TokIter<'a>>(iter: &mut Peekable<I>) -> Vec<AstNode> {
    if iter.next() != Some(&Token::LBrace) {
        panic!("Expected '{{' at start of block");
    }
    let mut block = Vec::new();
    while let Some(token) = iter.peek().cloned() {
        if *token == Token::RBrace {
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
    let Token::Ident(index_var) = token else {
        panic!("Expected identifier after 'for', found: {:?}", token);
    };
    token = iter.next().unwrap();
    trace!("Parsing for loop, second token: {:?}", token);
    let item_var = if token == &Token::Comma {
        token = iter.next().unwrap();
        trace!("Parsing for loop, item variable token: {:?}", token);
        let Token::Ident(item_var) = token else {
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
    if token != &Token::KeywordIn {
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
    if token == Some(&&Token::KeywordElse) {
        iter.next();
        else_block = parse_block(iter);
    }
    AstNode::IfExpression(Box::new(condition), block, else_block)
}

fn parse_primary_expression<'a, I: TokIter<'a>>(iter: &mut Peekable<I>) -> Option<AstNode> {
    let token = iter.peek()?;

    match token {
        Token::Operator(Operator::Minus) => {
            iter.next();
            let expr =
                parse_primary_expression(iter).expect("Expected expression after unary minus");
            Some(AstNode::BinaryOp(
                Box::new(AstNode::Literal(Value::Integer(0))),
                Operator::Minus,
                Box::new(expr),
            ))
        }
        Token::Literal(lit) => {
            iter.next();
            Some(AstNode::Literal(lit.clone()))
        }
        Token::Ident(ident) => {
            iter.next();
            trace!("Parsing identifier: {}", ident);
            if let Some(fcall) = parse_function_call(ident, iter) {
                Some(fcall)
            } else {
                Some(AstNode::Variable(ident.clone()))
            }
        }
        Token::LParen => {
            iter.next();
            let expr = parse_expression(iter).unwrap();
            let next = iter.peek().unwrap();
            if next != &&Token::RParen {
                panic!("Expected closing parenthesis, found: {:?}", next);
            }
            iter.next();
            Some(expr)
        }
        Token::LSquareParen => {
            iter.next();
            let elements = parse_list(iter, Token::Comma, Token::RSquareParen);
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
    while let Some(Token::Operator(op)) = iter.peek() {
        if op.precedence() < min_precedence {
            break;
        }
        let op = *op;
        iter.next();
        let mut right =
            parse_primary_expression(iter).expect("Expected an expression after operator");

        trace!(
            "Parsed right-hand side expression: {:?} after op, {:?}",
            right, op
        );

        while let Some(Token::Operator(next_op)) = iter.peek() {
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

    let statement = match token {
        Token::Ident(ident) => {
            iter.next();
            match *iter.peek().unwrap() {
                Token::Assign => {
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
        Token::KeywordFor => {
            iter.next();
            parse_for_loop(iter)
        }
        Token::KeywordIf => {
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
