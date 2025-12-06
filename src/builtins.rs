use std::{cell::RefCell, io::Write, rc::Rc};

use crate::tokenizer::{Range, Value, dbg_display};

pub fn fn_ok() -> ProgramFnRes {
    Ok(Value::Integer(1))
}

pub fn builtin_print(args: &[Value]) -> ProgramFnRes {
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

pub fn builtin_readfile(args: &[Value]) -> ProgramFnRes {
    let [Value::String(filename)] = &args else {
        return Err(format!("Expects (string), got {}", dbg_display(args)));
    };

    let content = std::fs::read_to_string(filename.as_ref())
        .map_err(|_| format!("Failed to read file: {}", filename))?;

    Ok(Value::String(content.trim().to_owned().into()))
}

pub fn builtin_split(args: &[Value]) -> ProgramFnRes {
    let [Value::String(s), Value::String(delim)] = &args else {
        return Err(format!(
            "Expects (string, string), got {}",
            dbg_display(args)
        ));
    };

    let parts: Vec<Value> = s
        .split(delim.as_ref())
        .map(|part| Value::String(Rc::from(part.to_owned())))
        .collect();

    Ok(Value::List(Rc::new(RefCell::new(parts))))
}

pub fn builtin_parseint(args: &[Value]) -> ProgramFnRes {
    let [Value::String(s)] = &args else {
        return Err(format!("Expects (string), got {}", dbg_display(args)));
    };

    let int_value = s
        .parse::<i64>()
        .map_err(|_| format!("Failed to parse integer from string: {}", s))?;

    Ok(Value::Integer(int_value))
}

pub fn builtin_substr(args: &[Value]) -> ProgramFnRes {
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
    Ok(Value::String(Rc::from(substring.to_owned())))
}

pub fn builtin_push(args: &[Value]) -> ProgramFnRes {
    let [target, value] = args else {
        return Err(format!("Expects (list, value), got {}", dbg_display(args)));
    };

    match target {
        Value::List(l) => {
            l.borrow_mut().push(value.clone());
            fn_ok()
        }
        _ => Err(format!(
            "Expects (list) as first argument, got {}",
            target.dbg_display()
        )),
    }
}

pub fn builtin_set(args: &[Value]) -> ProgramFnRes {
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

pub fn builtin_get(args: &[Value]) -> ProgramFnRes {
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

pub fn builtin_len(args: &[Value]) -> ProgramFnRes {
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

pub fn builtin_mod(args: &[Value]) -> ProgramFnRes {
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

pub fn builtin_range(args: &[Value]) -> ProgramFnRes {
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
