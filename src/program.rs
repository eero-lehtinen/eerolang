use foldhash::{HashMap, HashMapExt};
use std::{cell::RefCell, rc::Rc};

use crate::{
    ast_parser::{AstNode, AstNodeKind, fatal_generic},
    tokenizer::{Operator, Token, Value},
};

use crate::builtins::*;

#[allow(dead_code)]
pub struct Program {
    block: Rc<Vec<AstNode>>,
    tokens: Vec<Token>,
    vars: HashMap<Rc<str>, Value>,
    builtins: HashMap<String, ProgramFn>,
}

#[allow(dead_code)]
impl Program {
    pub fn new(block: Vec<AstNode>, tokens: Vec<Token>) -> Self {
        let builtins = HashMap::<String, ProgramFn>::from_iter(
            get_builtins()
                .iter()
                .map(|(name, func)| (name.to_string(), *func)),
        );

        Program {
            block: Rc::new(block),
            tokens,
            vars: HashMap::new(),
            builtins,
        }
    }

    fn fatal(&self, msg: &str, node: &AstNode) -> ! {
        let token = &self.tokens[node.token_idx];
        fatal_generic(msg, "Fatal error during program execution", token)
    }

    fn compute_expression<'a>(&'a mut self, expr: &'a AstNode) -> Value {
        match &expr.kind {
            AstNodeKind::Literal(lit) => lit.clone(),
            AstNodeKind::Variable(name) => self
                .vars
                .get(name)
                .unwrap_or_else(|| self.fatal(&format!("Undefined variable: {}", name), expr))
                .clone(),
            AstNodeKind::FunctionCall(name, args) => self.call_function(name, args, expr),
            AstNodeKind::List(list) => {
                let values = list
                    .iter()
                    .map(|elem| self.compute_expression(elem))
                    .collect::<Vec<_>>();
                Value::List(Rc::new(RefCell::new(values)))
            }
            AstNodeKind::BinaryOp(left, op, right) => {
                let mut left_val = self.compute_expression(left);
                let mut right_val = self.compute_expression(right);

                macro_rules! unsupported {
                    () => {
                        self.fatal(
                            &format!(
                                "Cannot apply operator {:?} to operands (left: {:?}, right: {:?})",
                                op, left_val, right_val,
                            ),
                            expr,
                        )
                    };
                }

                if let (Value::String(l), Value::String(r), Operator::Plus) =
                    (&left_val, &right_val, op)
                {
                    return Value::String(Rc::from([l.as_ref(), r.as_ref()].concat()));
                }
                if let (Value::String(l), Value::String(r)) = (&left_val, &right_val) {
                    return match op {
                        Operator::Plus => {
                            Value::String(Rc::from([l.as_ref(), r.as_ref()].concat()))
                        }
                        Operator::Eq => Value::Integer((l == r) as i64),
                        Operator::Neq => Value::Integer((l != r) as i64),
                        _ => unsupported!(),
                    };
                }

                if let (Value::Integer(l), Value::Integer(r)) = (&left_val, &right_val) {
                    return Value::Integer(match op {
                        Operator::Plus => l + r,
                        Operator::Minus => l - r,
                        Operator::Multiply => l * r,
                        Operator::Divide => l / r,
                        Operator::Lt => (l < r) as i64,
                        Operator::Gt => (l > r) as i64,
                        Operator::Lte => (l <= r) as i64,
                        Operator::Gte => (l >= r) as i64,
                        Operator::Eq => (l == r) as i64,
                        Operator::Neq => (l != r) as i64,
                    });
                }

                // Promote to float if both weren't integers
                if let Value::Integer(i) = &right_val {
                    right_val = Value::Float(*i as f64);
                };
                if let Value::Integer(i) = &left_val {
                    left_val = Value::Float(*i as f64);
                };

                if let (Value::Float(l), Value::Float(r)) = (&left_val, &right_val) {
                    return match op {
                        Operator::Plus => Value::Float(l + r),
                        Operator::Minus => Value::Float(l - r),
                        Operator::Multiply => Value::Float(l * r),
                        Operator::Divide => Value::Float(l / r),
                        Operator::Lt => Value::Integer((l < r) as i64),
                        Operator::Gt => Value::Integer((l > r) as i64),
                        Operator::Lte => Value::Integer((l <= r) as i64),
                        Operator::Gte => Value::Integer((l >= r) as i64),
                        Operator::Eq => Value::Integer((l == r) as i64),
                        Operator::Neq => Value::Integer((l != r) as i64),
                    };
                }

                unsupported!();
            }
            _ => self.fatal(
                &format!("Unexpected AST node in expression: {:#?}", expr),
                expr,
            ),
        }
    }

    fn call_function(&mut self, name: &str, args: &[AstNode], node: &AstNode) -> Value {
        let mut arg_values = [const { Value::Integer(0) }; 10];
        if args.len() > arg_values.len() {
            self.fatal(
                &format!(
                    "Function {} called with too many arguments (max={})",
                    name,
                    arg_values.len()
                ),
                node,
            );
        }
        for (v, a) in arg_values.iter_mut().zip(args.iter()) {
            *v = self.compute_expression(a);
        }
        if let Some(func) = self.builtins.get(name) {
            match func(&mut arg_values[..args.len()]) {
                Ok(val) => val,
                Err(err) => self.fatal(&format!("Error in function {}: {}", name, err), node),
            }
        } else {
            self.fatal(&format!("Undefined function: {}", name), node);
        }
    }

    fn execute_block(&mut self, block: &[AstNode]) {
        for node in block.iter() {
            match &node.kind {
                AstNodeKind::Assign(var, expr) => {
                    let value = self.compute_expression(expr);
                    self.vars.insert(Rc::clone(var), value.clone());
                }
                AstNodeKind::FunctionCall(name, args) => {
                    self.call_function(name, args, node);
                }
                AstNodeKind::ForLoop(index_var, item_var, collection, body) => {
                    let collection = self.compute_expression(collection);

                    let index_key = if index_var.as_ref() != "_" {
                        self.vars.insert(Rc::clone(index_var), Value::Integer(0));
                        Some(index_var)
                    } else {
                        None
                    };
                    let item_key = if let Some(item_var) = item_var
                        && item_var.as_ref() != "_"
                    {
                        self.vars.insert(Rc::clone(item_var), Value::Integer(0));
                        Some(item_var)
                    } else {
                        None
                    };

                    match collection {
                        Value::List(l) => {
                            for (i, elem) in l.borrow().iter().enumerate() {
                                if let Some(index_key) = index_key {
                                    *self.vars.get_mut(index_key).unwrap() =
                                        Value::Integer(i as i64);
                                }
                                if let Some(item_key) = item_key {
                                    *self.vars.get_mut(item_key).unwrap() = elem.clone();
                                }
                                self.execute_block(body);
                            }
                        }
                        Value::Range(r) => {
                            for i in r.start..r.end {
                                if let Some(index_key) = index_key {
                                    *self.vars.get_mut(index_key).unwrap() = Value::Integer(i);
                                }
                                self.execute_block(body);
                            }
                        }
                        _ => self.fatal(
                            &format!("For loop expects a list or range, got {:?}", collection),
                            node,
                        ),
                    };
                }
                AstNodeKind::IfStatement(condition, block, else_block) => {
                    let cond_value = self.compute_expression(condition);
                    let cond_true = match cond_value {
                        Value::Integer(i) => i != 0,
                        _ => self.fatal(
                            &format!(
                                "If condition must evaluate to an integer, got {:?}",
                                cond_value
                            ),
                            node,
                        ),
                    };
                    if cond_true {
                        self.execute_block(block);
                    } else {
                        self.execute_block(else_block);
                    }
                }
                _ => {
                    self.fatal(
                        &format!("Unexpected AST node during execution: {:#?}", node),
                        node,
                    );
                }
            }
        }
    }

    pub fn execute(&mut self) {
        let block = Rc::clone(&self.block);
        self.execute_block(&block);
    }
}
