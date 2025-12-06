use foldhash::{HashMap, HashMapExt};
use std::{cell::RefCell, io::Write, rc::Rc};

use crate::{
    ast_parser::{AstNode, AstNodeKind, fatal_generic},
    tokenizer::{Operator, Range, Token, Value, dbg_display},
};

fn fn_ok() -> ProgramFnRes {
    Ok(Value::Integer(1))
}

fn builtin_print(args: &mut [Value]) -> ProgramFnRes {
    let mut w = std::io::stdout();
    for (i, arg) in args.iter().enumerate() {
        match arg {
            Value::Integer(i) => write!(&mut w, "{}", i).unwrap(),
            Value::Float(f) => write!(&mut w, "{}", f).unwrap(),
            Value::String(s) => write!(&mut w, "{}", s).unwrap(),
            Value::Range(r) => {
                write!(&mut w, "{}..{}", r.start, r.end).unwrap();
            }
            Value::List(l) => {
                write!(&mut w, "[").unwrap();
                for (j, item) in l.borrow().iter().enumerate() {
                    match item {
                        Value::Integer(ii) => write!(&mut w, "{}", ii).unwrap(),
                        Value::Float(ff) => write!(&mut w, "{}", ff).unwrap(),
                        Value::String(ss) => write!(&mut w, "\"{}\"", ss).unwrap(),
                        Value::List(_) => write!(&mut w, "<nested list>").unwrap(),
                        Value::Range(r) => write!(&mut w, "(range {},{})", r.start, r.end).unwrap(),
                    }
                    if j < l.borrow().len() - 1 {
                        print!(", ");
                    }
                }
                write!(&mut w, "]").unwrap();
            }
        }
        if i < args.len() - 1 {
            write!(&mut w, " ").unwrap();
        }
    }
    writeln!(&mut w).unwrap();
    w.flush().unwrap();
    fn_ok()
}

fn builtin_readfile(args: &mut [Value]) -> ProgramFnRes {
    let [Value::String(filename)] = &args else {
        return Err(format!("Expects (string), got {}", dbg_display(args)));
    };

    let content = std::fs::read_to_string(filename.as_ref())
        .map_err(|_| format!("Failed to read file: {}", filename))?;

    Ok(Value::String(content.trim().into()))
}

fn builtin_split(args: &mut [Value]) -> ProgramFnRes {
    let [Value::String(s), Value::String(delim)] = &args else {
        return Err(format!(
            "Expects (string, string), got {}",
            dbg_display(args)
        ));
    };

    let parts: Vec<Value> = s
        .split(delim.as_ref())
        .map(|part| Value::String(Rc::from(part)))
        .collect();

    Ok(Value::List(Rc::new(RefCell::new(parts))))
}

fn builtin_parseint(args: &mut [Value]) -> ProgramFnRes {
    let [Value::String(s)] = &args else {
        return Err(format!("Expects (string), got {}", dbg_display(args)));
    };

    let int_value = s
        .parse::<i64>()
        .map_err(|_| format!("Failed to parse integer from string: {}", s))?;

    Ok(Value::Integer(int_value))
}

fn builtin_substr(args: &mut [Value]) -> ProgramFnRes {
    if args.len() < 2 || args.len() > 3 {
        return Err(format!(
            "Expects (string, int, opt int), got {}",
            dbg_display(args)
        ));
    }
    let string = match &args[0] {
        Value::String(s) => s,
        _ => {
            return Err(format!(
                "Expects (string) as first argument, got {}",
                args[0].dbg_display()
            ));
        }
    };
    let start = match &args[1] {
        Value::Integer(i) => *i,
        _ => {
            return Err(format!(
                "Expects (int) as second argument, got {}",
                args[1].dbg_display()
            ));
        }
    };
    let mut end = if args.get(2).is_some() {
        match &args[2] {
            Value::Integer(i) => *i,
            _ => {
                return Err(format!(
                    "Expects (int) as third argument, got {}",
                    args[2].dbg_display()
                ));
            }
        }
    } else {
        string.len() as i64
    };

    if start < 0 || start > end {
        return Err(format!(
            "Expects start to be non-negative and less than end, got start: {}, end: {}",
            start, end
        ));
    }
    if end < 0 {
        end = string.len() as i64 - end.abs();
    }
    end = end.min(string.len() as i64);

    let substring = &string[start as usize..end as usize];
    Ok(Value::String(Rc::from(substring)))
}

fn builtin_set(args: &mut [Value]) -> ProgramFnRes {
    let [target, index, value] = args else {
        return Err(format!(
            "Expects (list, int, value), got {}",
            dbg_display(args)
        ));
    };

    let index = match &index {
        Value::Integer(i) => *i as usize,
        _ => {
            return Err(format!(
                "Expects (int) as second argument, got {}",
                index.dbg_display()
            ));
        }
    };

    match target {
        Value::List(l) => {
            if index >= l.borrow().len() {
                return Err(format!(
                    "Index out of bounds (length: {}, index: {})",
                    l.borrow().len(),
                    index
                ));
            }
            l.borrow_mut()[index] = value.clone();
            fn_ok()
        }
        _ => Err(format!(
            "Expects (list) as first argument, got {}",
            target.dbg_display()
        )),
    }
}

fn builtin_get(args: &mut [Value]) -> ProgramFnRes {
    let [target, index] = args else {
        return Err(format!(
            "Expects (list/string, int), got {}",
            dbg_display(args)
        ));
    };

    let index = match index {
        Value::Integer(i) => *i as usize,
        _ => {
            return Err(format!(
                "Expects (int) as second argument, got {}",
                index.dbg_display()
            ));
        }
    };

    match target {
        Value::List(l) => {
            if index >= l.borrow().len() {
                return Err(format!(
                    "Index out of bounds (length: {}, index: {})",
                    l.borrow().len(),
                    index
                ));
            }
            Ok(l.borrow()[index].clone())
        }
        Value::String(s) => {
            if index >= s.len() {
                return Err(format!(
                    "Index out of bounds (length: {}, index: {})",
                    s.len(),
                    index
                ));
            }
            Ok(Value::String(Rc::from(
                s.chars().nth(index).unwrap().to_string(),
            )))
        }
        _ => Err(format!(
            "Expects (list/string) as first argument, got {}",
            target.dbg_display()
        )),
    }
}

fn builtin_len(args: &mut [Value]) -> ProgramFnRes {
    let [len] = args else {
        return Err(format!("Expects (list/string), got {}", dbg_display(args)));
    };

    let len = match &len {
        Value::String(s) => s.len() as i64,
        Value::List(l) => l.borrow().len() as i64,
        _ => return Err(format!("Expects (list/string), got {}", len.dbg_display())),
    };

    Ok(Value::Integer(len))
}

fn builtin_mod(args: &mut [Value]) -> ProgramFnRes {
    let [a, b] = args else {
        return Err(format!("Expects (int, int), got {}", dbg_display(args)));
    };

    let a = match a {
        Value::Integer(i) => *i,
        _ => {
            return Err(format!(
                "Expects (int) as first argument, got {}",
                a.dbg_display()
            ));
        }
    };

    let b = match b {
        Value::Integer(i) => *i,
        _ => {
            return Err(format!(
                "Expects (int) as second argument, got {}",
                b.dbg_display()
            ));
        }
    };

    Ok(Value::Integer(a % b))
}

fn builtin_range(args: &mut [Value]) -> ProgramFnRes {
    if args.is_empty() || args.len() > 2 {
        return Err(format!("Expects (int, opt int), got {}", dbg_display(args)));
    };

    let mut start = match &args[0] {
        Value::Integer(i) => *i,
        _ => {
            return Err(format!(
                "Expects (int) as first argument, got {}",
                args[0].dbg_display()
            ));
        }
    };

    let end = match args.get(1) {
        Some(Value::Integer(i)) => *i,
        None => {
            let tmp = start;
            start = 0;
            tmp
        }
        _ => {
            return Err(format!(
                "Expects (opt int) as second argument, got {}",
                args[1].dbg_display()
            ));
        }
    };

    Ok(Value::Range(Box::new(Range { start, end })))
}

pub type ProgramFnRes = Result<Value, String>;
pub type ProgramFn = fn(&mut [Value]) -> ProgramFnRes;

pub struct Program {
    block: Rc<Vec<AstNode>>,
    tokens: Vec<Token>,
    vars: HashMap<Rc<str>, Value>,
    builtins: HashMap<String, ProgramFn>,
}

impl Program {
    pub fn new(block: Vec<AstNode>, tokens: Vec<Token>) -> Self {
        let builtins: [(_, ProgramFn); _] = [
            ("print", builtin_print),
            ("readfile", builtin_readfile),
            ("split", builtin_split),
            ("parseint", builtin_parseint),
            ("substr", builtin_substr),
            ("len", builtin_len),
            ("get", builtin_get),
            ("set", builtin_set),
            ("mod", builtin_mod),
            ("range", builtin_range),
        ];
        let builtins = HashMap::<String, ProgramFn>::from_iter(
            builtins.map(|(name, func)| (name.to_owned(), func)),
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
                AstNodeKind::IfExpression(condition, block, else_block) => {
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
