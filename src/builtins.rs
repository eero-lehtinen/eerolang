use std::{cell::RefCell, io::Write, rc::Rc};

use crate::tokenizer::{Range, Value, arg_display};

macro_rules! arg_bail {
    ($expected:expr, $args:expr) => {{
        return Err(format!(
            "Expects ({}), got {}",
            $expected,
            arg_display($args)
        ));
    }};
}

macro_rules! out_of_bounds_bail {
    ($length:expr, $index:expr) => {{
        return Err(format!(
            "Index out of bounds (length: {}, index: {})",
            $length, $index
        ));
    }};
}

macro_rules! fn_ok {
    () => {
        Ok(Value::Integer(1))
    };
}

pub fn builtin_print(args: &[Value]) -> ProgramFnRes {
    let mut w = std::io::stdout();
    for (i, arg) in args.iter().enumerate() {
        macro_rules! print_inner {
            ($item:expr) => {
                match $item {
                    Value::Integer(ii) => write!(&mut w, "{}", ii).unwrap(),
                    Value::Float(ff) => write!(&mut w, "{}", ff).unwrap(),
                    Value::String(ss) => write!(&mut w, "{}", ss).unwrap(),
                    Value::List(_) => write!(&mut w, "(list...)").unwrap(),
                    Value::Map(_) => write!(&mut w, "(map...)").unwrap(),
                    Value::Range(r) => write!(&mut w, "(range {},{})", r.start, r.end).unwrap(),
                }
            };
        }

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
                    print_inner!(item);
                    if j < l.borrow().len() - 1 {
                        print!(", ");
                    }
                }
                write!(&mut w, "]").unwrap();
            }
            Value::Map(m) => {
                write!(&mut w, "{{").unwrap();
                let map_ref = m.borrow();
                for (j, (key, value)) in map_ref.iter().enumerate() {
                    write!(&mut w, "\"{}\": ", key).unwrap();
                    print_inner!(value);
                    if j < map_ref.len() - 1 {
                        write!(&mut w, ", ").unwrap();
                    }
                }
                write!(&mut w, "}}").unwrap();
            }
        }
        if i < args.len() - 1 {
            write!(&mut w, " ").unwrap();
        }
    }
    writeln!(&mut w).unwrap();
    w.flush().unwrap();
    fn_ok!()
}

pub fn builtin_readfile(args: &[Value]) -> ProgramFnRes {
    let [Value::String(filename)] = &args else {
        arg_bail!("string", args);
    };

    let content = std::fs::read_to_string(filename.as_ref())
        .map_err(|_| format!("Failed to read file: {}", filename))?;

    Ok(Value::String(content.trim().to_owned().into()))
}

pub fn builtin_split(args: &[Value]) -> ProgramFnRes {
    let [Value::String(s), Value::String(delim)] = &args else {
        arg_bail!("string, string", args);
    };

    let parts: Vec<Value> = s
        .split(delim.as_ref())
        .map(|part| Value::String(Rc::from(part.to_owned())))
        .collect();

    Ok(Value::List(Rc::new(RefCell::new(parts))))
}

pub fn builtin_parseint(args: &[Value]) -> ProgramFnRes {
    let [Value::String(s)] = &args else {
        arg_bail!("string", args);
    };

    let int_value = s
        .parse::<i64>()
        .map_err(|_| format!("Failed to parse integer from string: {}", s))?;

    Ok(Value::Integer(int_value))
}

pub fn builtin_substr(args: &[Value]) -> ProgramFnRes {
    if args.len() < 2 || args.len() > 3 {
        arg_bail!("string, int, opt int", args);
    }
    let string = match &args[0] {
        Value::String(s) => s,
        _ => arg_bail!("string, int, opt int", args),
    };
    let start = match &args[1] {
        Value::Integer(i) => *i,
        _ => arg_bail!("string, int, opt int", args),
    };
    let mut end = if args.get(2).is_some() {
        match &args[2] {
            Value::Integer(i) => *i,
            _ => arg_bail!("string, int, opt int", args),
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
    Ok(Value::String(Rc::from(substring.to_owned())))
}

#[inline]
pub fn builtin_push(args: &[Value]) -> ProgramFnRes {
    let [target, value] = args else {
        arg_bail!("list, value", args);
    };

    match target {
        Value::List(l) => {
            l.borrow_mut().push(value.clone());
            fn_ok!()
        }
        _ => arg_bail!("list, value", args),
    }
}

pub fn builtin_set(args: &[Value]) -> ProgramFnRes {
    let [target, index_or_key, value] = args else {
        arg_bail!("list/map, int if list/string if map, value", args);
    };

    match target {
        Value::List(l) => {
            let index = match &index_or_key {
                Value::Integer(i) => *i as usize,
                _ => arg_bail!("list/map, int if list/string if map, value", args),
            };
            if index >= l.borrow().len() {
                out_of_bounds_bail!(l.borrow().len(), index);
            }
            l.borrow_mut()[index] = value.clone();
            fn_ok!()
        }
        _ => arg_bail!("list/map, int if list/string if map, value", args),
    }
}

#[inline]
pub fn builtin_get(args: &[Value]) -> ProgramFnRes {
    let [target, index] = args else {
        arg_bail!("list/string/map, int/string if map", args);
    };

    let index = match index {
        Value::Integer(i) => *i as usize,
        _ => arg_bail!("list/string/map, int/string if map", args),
    };

    match target {
        Value::List(l) => {
            if index >= l.borrow().len() {
                out_of_bounds_bail!(l.borrow().len(), index);
            }
            Ok(l.borrow()[index].clone())
        }
        Value::String(s) => {
            if index >= s.len() {
                out_of_bounds_bail!(s.len(), index);
            }
            Ok(Value::String(Rc::from(
                s.chars().nth(index).unwrap().to_string(),
            )))
        }
        _ => arg_bail!("list/string/map, int/string if map", args),
    }
}

pub fn builtin_len(args: &[Value]) -> ProgramFnRes {
    let [len] = args else {
        arg_bail!("list/string/map", args);
    };

    let len = match &len {
        Value::String(s) => s.len() as i64,
        Value::List(l) => l.borrow().len() as i64,
        _ => arg_bail!("list/string/map", args),
    };

    Ok(Value::Integer(len))
}

pub fn builtin_mod(args: &[Value]) -> ProgramFnRes {
    let [a, b] = args else {
        arg_bail!("int, int", args);
    };

    let a = match a {
        Value::Integer(i) => *i,
        _ => arg_bail!("int, int", args),
    };

    let b = match b {
        Value::Integer(i) => *i,
        _ => arg_bail!("int, int", args),
    };

    Ok(Value::Integer(a % b))
}

pub fn builtin_range(args: &[Value]) -> ProgramFnRes {
    if args.is_empty() || args.len() > 2 {
        arg_bail!("int, opt int", args);
    };

    let mut start = match &args[0] {
        Value::Integer(i) => *i,
        _ => arg_bail!("int, opt int", args),
    };

    let end = match args.get(1) {
        Some(Value::Integer(i)) => *i,
        None => {
            let tmp = start;
            start = 0;
            tmp
        }
        _ => arg_bail!("int, opt int", args),
    };

    Ok(Value::Range(Box::new(Range { start, end })))
}

pub type ProgramFnRes = Result<Value, String>;
pub type ProgramFn = fn(&[Value]) -> ProgramFnRes;

pub fn all_builtins() -> Vec<(&'static str, ProgramFn)> {
    vec![
        ("print", builtin_print),
        ("readfile", builtin_readfile),
        ("split", builtin_split),
        ("parseint", builtin_parseint),
        ("substr", builtin_substr),
        ("push", builtin_push),
        ("set", builtin_set),
        ("get", builtin_get),
        ("len", builtin_len),
        ("mod", builtin_mod),
        ("range", builtin_range),
    ]
}
