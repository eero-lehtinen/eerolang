use std::{cell::RefCell, collections::HashMap, io::Write, rc::Rc};

use log::trace;

use crate::{
    ast_parser::AstNode,
    tokenizer::{Operator, Value},
};

fn builtin_print(args: &mut [Value]) -> Option<Value> {
    let mut w = std::io::stdout();
    for (i, arg) in args.iter().enumerate() {
        match arg {
            Value::Integer(i) => write!(&mut w, "{}", i).unwrap(),
            Value::Float(f) => write!(&mut w, "{}", f).unwrap(),
            Value::String(s) => write!(&mut w, "{}", s).unwrap(),
            Value::Range(start, end) => {
                write!(&mut w, "{}..{}", start, end).unwrap();
            }
            Value::List(l) => {
                write!(&mut w, "[").unwrap();
                for (j, item) in l.borrow().iter().enumerate() {
                    match item {
                        Value::Integer(ii) => write!(&mut w, "{}", ii).unwrap(),
                        Value::Float(ff) => write!(&mut w, "{}", ff).unwrap(),
                        Value::String(ss) => write!(&mut w, "\"{}\"", ss).unwrap(),
                        Value::List(_) => write!(&mut w, "<nested list>").unwrap(),
                        Value::Range(s, e) => write!(&mut w, "{}..{}", s, e).unwrap(),
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
    None
}

fn builtin_readfile(args: &mut [Value]) -> Option<Value> {
    let [Value::String(filename)] = &args else {
        panic!("readfile expects (string), got {:?}", args)
    };

    let content = std::fs::read_to_string(filename.as_ref())
        .unwrap_or_else(|_| panic!("Failed to read file: {}", filename));

    Some(Value::String(content.trim().into()))
}

fn builtin_split(args: &mut [Value]) -> Option<Value> {
    let [Value::String(s), Value::String(delim)] = &args else {
        panic!("split expects (string, string), got {:?}", args)
    };

    trace!("Splitting string '{}' by delimiter '{}'", s, delim);

    let parts: Vec<Value> = s
        .split(delim.as_ref())
        .map(|part| Value::String(Rc::from(part)))
        .collect();

    Some(Value::List(Rc::new(RefCell::new(parts))))
}

fn builtin_parseint(args: &mut [Value]) -> Option<Value> {
    let [Value::String(s)] = &args else {
        panic!("parseint expects (string), got {:?}", args)
    };

    let int_value = s
        .parse::<i64>()
        .unwrap_or_else(|_| panic!("Failed to parse integer from string: {}", s));

    Some(Value::Integer(int_value))
}

fn builtin_substr(args: &mut [Value]) -> Option<Value> {
    if args.len() < 2 || args.len() > 3 {
        panic!("substr expects (string, int, opt int), got {:?}", args)
    }
    let string = match &args[0] {
        Value::String(s) => s,
        _ => panic!("substr expects first argument to be string"),
    };
    let start = match &args[1] {
        Value::Integer(i) => *i,
        _ => panic!("substr expects second argument to be integer"),
    };
    let mut end = if args.get(2).is_some() {
        match &args[2] {
            Value::Integer(i) => *i,
            _ => panic!("substr expects third argument to be integer"),
        }
    } else {
        string.len() as i64
    };

    if start < 0 || start > end {
        panic!("substr indices out of bounds");
    }
    if end < 0 {
        end = string.len() as i64 - end.abs();
    }
    end = end.min(string.len() as i64);

    let substring = &string[start as usize..end as usize];
    Some(Value::String(Rc::from(substring)))
}

fn builtin_set(args: &mut [Value]) -> Option<Value> {
    let [target, index, value] = args else {
        panic!("set expects (list/string, int, value), got {:?}", args)
    };

    let index = match &index {
        Value::Integer(i) => *i as usize,
        _ => panic!("set expects second argument to be integer index"),
    };

    match target {
        Value::List(l) => {
            if index >= l.borrow().len() {
                panic!("set index out of bounds");
            }
            l.borrow_mut()[index] = value.clone();
            None
        }
        Value::String(s) => {
            if index >= s.len() {
                panic!("set index out of bounds");
            }
            let Value::String(new_char_str) = value else {
                panic!("set value for string must be a single character string");
            };
            let new_char = new_char_str
                .chars()
                .next()
                .expect("set value for string must be a single character string");
            let mut chars: Vec<char> = s.chars().collect();
            chars[index] = new_char;
            let new_string: String = chars.into_iter().collect();
            *s = Rc::from(new_string);
            None
        }
        _ => panic!("set expects (list, int, value), got {:?}", args),
    }
}

fn builtin_get(args: &mut [Value]) -> Option<Value> {
    let [target, index] = args else {
        panic!("get expects (list/string, int), got {:?}", args)
    };

    let index = match index {
        Value::Integer(i) => *i as usize,
        _ => panic!("get expects (list/string, int), got {:?}", args),
    };

    match target {
        Value::List(l) => {
            if index >= l.borrow().len() {
                panic!("get index out of bounds");
            }
            Some(l.borrow()[index].clone())
        }
        Value::String(s) => {
            if index >= s.len() {
                panic!("get index out of bounds");
            }
            Some(Value::String(Rc::from(
                s.chars().nth(index).unwrap().to_string(),
            )))
        }
        _ => panic!("get expects (list/string, int), got {:?}", args),
    }
}

fn builtin_len(args: &mut [Value]) -> Option<Value> {
    let [len] = args else {
        panic!("len expects (list/string), got {:?}", args)
    };

    let len = match &len {
        Value::String(s) => s.len() as i64,
        Value::List(l) => l.borrow().len() as i64,
        _ => panic!("len expects (list/string), got {:?}", args),
    };

    Some(Value::Integer(len))
}

fn builtin_mod(args: &mut [Value]) -> Option<Value> {
    let [a, b] = args else {
        panic!("mod expects (int, int), got {:?}", args)
    };

    let a = match a {
        Value::Integer(i) => *i,
        _ => panic!("mod expects (int, int), got {:?}", args),
    };

    let b = match b {
        Value::Integer(i) => *i,
        _ => panic!("mod expects (int, int), got {:?}", args),
    };

    Some(Value::Integer(a % b))
}

fn builtin_range(args: &mut [Value]) -> Option<Value> {
    if args.is_empty() || args.len() > 2 {
        panic!("range expects (int, opt int), got {:?}", args)
    };

    let mut start = match &args[0] {
        Value::Integer(i) => *i,
        _ => panic!("range expects (int, opt int), got {:?}", args),
    };

    let end = match args.get(1) {
        Some(Value::Integer(i)) => *i,
        None => {
            let tmp = start;
            start = 0;
            tmp
        }
        _ => panic!("range expects (int, opt int), got {:?}", args),
    };

    Some(Value::Range(start, end))
}

pub type ProgramFn = fn(&mut [Value]) -> Option<Value>;

pub struct Program {
    block: Rc<Vec<AstNode>>,
    vars: HashMap<Rc<str>, Value>,
    builtins: HashMap<String, ProgramFn>,
}

impl Program {
    pub fn new(block: Vec<AstNode>) -> Self {
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
        let builtins = HashMap::<String, ProgramFn>::from(
            builtins.map(|(name, func)| (name.to_owned(), func)),
        );

        Program {
            block: Rc::new(block),
            vars: HashMap::new(),
            builtins,
        }
    }

    fn compute_expression<'a>(&'a mut self, expr: &'a AstNode) -> Value {
        match expr {
            AstNode::Literal(lit) => lit.clone(),
            AstNode::Variable(name) => self
                .vars
                .get(name.as_str())
                .unwrap_or_else(|| panic!("Undefined variable: {}", name))
                .clone(),
            AstNode::FunctionCall(name, args) => self
                .call_function(name, args)
                .expect("Function did not return a value"),
            AstNode::List(list) => {
                let values = list
                    .iter()
                    .map(|elem| self.compute_expression(elem))
                    .collect::<Vec<_>>();
                Value::List(Rc::new(RefCell::new(values)))
            }
            AstNode::BinaryOp(left, op, right) => {
                let mut left_val = self.compute_expression(left);
                let mut right_val = self.compute_expression(right);
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
                        _ => panic!("Cannot apply operator {:?} to strings", op),
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

                if let (Value::Float(l), Value::Float(r)) = (left_val, right_val) {
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

                panic!("Unsupported operand types for binary operation");
            }
            _ => panic!("Unsupported expression type"),
        }
    }

    fn call_function(&mut self, name: &str, args: &[AstNode]) -> Option<Value> {
        let mut arg_values = args
            .iter()
            .map(|arg| self.compute_expression(arg))
            .collect::<Vec<_>>();
        if let Some(func) = self.builtins.get(name) {
            func(&mut arg_values)
        } else {
            panic!("Undefined function: {}", name);
        }
    }

    fn execute_block(&mut self, block: &[AstNode]) {
        for node in block.iter() {
            match node {
                AstNode::Assign(var, expr) => {
                    trace!("Assigning to variable: {}", var);
                    let value = self.compute_expression(expr);
                    self.vars.insert(Rc::from(var.clone()), value.clone());
                }
                AstNode::FunctionCall(name, args) => {
                    trace!("Calling function: {}", name);
                    self.call_function(name, args);
                }
                AstNode::ForLoop(index_var, item_var, collection, body) => {
                    let collection = self.compute_expression(collection);

                    let index_key = Rc::from(index_var.as_str());
                    let item_key = if let Some(v) = item_var {
                        Rc::from(v.as_str())
                    } else {
                        Rc::from("_INTERNAL_UNUSED")
                    };
                    self.vars.insert(Rc::clone(&index_key), Value::Integer(0));
                    self.vars.insert(Rc::clone(&item_key), Value::Integer(0));

                    match collection {
                        Value::List(l) => {
                            for (i, elem) in l.borrow().iter().enumerate() {
                                *self.vars.get_mut(&index_key).unwrap() = Value::Integer(i as i64);
                                *self.vars.get_mut(&item_key).unwrap() = elem.clone();
                                self.execute_block(body);
                            }
                        }
                        Value::Range(start, end) => {
                            for i in start..end {
                                *self.vars.get_mut(&index_key).unwrap() = Value::Integer(i);
                                self.execute_block(body);
                            }
                        }
                        _ => panic!("For loop expects a list or range"),
                    };
                }
                AstNode::IfExpression(condition, block, else_block) => {
                    let cond_value = self.compute_expression(condition);
                    let cond_true = match cond_value {
                        Value::Integer(i) => i != 0,
                        _ => panic!("If condition must evaluate to an integer"),
                    };
                    if cond_true {
                        self.execute_block(block);
                    } else {
                        self.execute_block(else_block);
                    }
                }
                _ => {
                    panic!("Unexpected AST node during execution: {:#?}", node);
                }
            }
        }
    }

    pub fn execute(&mut self) {
        let block = Rc::clone(&self.block);
        self.execute_block(&block);
    }
}
