use std::{cell::RefCell, io::Write, rc::Rc};

use foldhash::HashMap;

use crate::tokenizer::{MapKey, MapValue, Range, Value, arg_display};

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
                    Value::String(ss) => write!(&mut w, "\"{}\"", ss).unwrap(),
                    Value::List(_) => write!(&mut w, "(list...)").unwrap(),
                    Value::Map(_) => write!(&mut w, "(map...)").unwrap(),
                    Value::Range(r) => write!(&mut w, "(range {}-{})", r.start, r.end).unwrap(),
                }
            };
        }

        match arg {
            Value::Integer(i) => write!(&mut w, "{}", i).unwrap(),
            Value::Float(f) => write!(&mut w, "{}", f).unwrap(),
            Value::String(s) => write!(&mut w, "{}", s).unwrap(),
            Value::Range(r) => {
                write!(&mut w, "{}-{}", r.start, r.end).unwrap();
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
                for (j, (key, value)) in map_ref.inner.iter().enumerate() {
                    write!(&mut w, "{}: ", key.dbg_display()).unwrap();
                    print_inner!(value);
                    if j < map_ref.inner.len() - 1 {
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

const READFILE_ARGS: u32 = 1;
pub fn builtin_readfile(args: &[Value]) -> ProgramFnRes {
    let [Value::String(filename)] = &args else {
        arg_bail!("string", args);
    };

    let content = std::fs::read_to_string(filename.as_ref())
        .map_err(|_| format!("Failed to read file: {}", filename))?;

    Ok(Value::String(content.trim().to_owned().into()))
}

const SPLIT_ARGS: u32 = 2;
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

const INT_ARGS: u32 = 1;
pub fn builtin_int(args: &[Value]) -> ProgramFnRes {
    let [arg] = args else {
        arg_bail!("string/float/int", args);
    };

    match arg {
        Value::String(s) => {
            let int_value = s
                .parse::<i64>()
                .map_err(|_| format!("Failed to parse integer from string: {}", s))?;
            Ok(Value::Integer(int_value))
        }
        Value::Float(f) => Ok(Value::Integer(*f as i64)),
        Value::Integer(i) => Ok(Value::Integer(*i)),
        _ => arg_bail!("string/float", args),
    }
}

const FLOAT_ARGS: u32 = 1;
pub fn builtin_float(args: &[Value]) -> ProgramFnRes {
    let [arg] = args else {
        arg_bail!("string/int", args);
    };

    match arg {
        Value::String(s) => {
            let float_value = s
                .parse::<f64>()
                .map_err(|_| format!("Failed to parse float from string: {}", s))?;
            Ok(Value::Float(float_value))
        }
        Value::Integer(i) => Ok(Value::Float(*i as f64)),
        Value::Float(f) => Ok(Value::Float(*f)),
        _ => arg_bail!("string/int", args),
    }
}

fn write_str(w: &mut impl Write, value: &Value) {
    match value {
        Value::Integer(i) => write!(w, "{}", i).unwrap(),
        Value::Float(f) => write!(w, "{}", f).unwrap(),
        Value::String(s) => write!(w, "{}", s).unwrap(),
        Value::List(elems) => {
            for (i, v) in elems.borrow().iter().enumerate() {
                if i > 0 {
                    write!(w, ",").unwrap();
                }
                write_str(w, v);
            }
        }
        Value::Map(map) => {
            let map_ref = map.borrow();
            let mut first = true;
            for (key, value) in map_ref.inner.iter() {
                if !first {
                    write!(w, ",").unwrap();
                }
                first = false;
                write!(w, "{}", key.dbg_display()).unwrap();
                write!(w, ":").unwrap();
                write_str(w, value);
            }
        }
        Value::Range(r) => write!(w, "{}-{}", r.start, r.end).unwrap(),
    }
}

const STRING_ARGS: u32 = 1;
pub fn builtin_string(args: &[Value]) -> ProgramFnRes {
    let [arg] = args else {
        arg_bail!("value", args);
    };
    let mut w = Vec::new();
    write_str(&mut w, arg);

    let s = String::from_utf8(w).unwrap();

    Ok(Value::String(Rc::from(s)))
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

pub fn builtin_list(args: &[Value]) -> ProgramFnRes {
    let list = args.to_vec();
    Ok(Value::List(Rc::new(RefCell::new(list))))
}

pub fn builtin_map(args: &[Value]) -> ProgramFnRes {
    let values = args
        .iter()
        .map(|arg| {
            let Value::List(pair) = arg else {
                arg_bail!("all arguments to be pairs [string/int key, value]", args);
            };
            let [key, value] = &pair.borrow()[..] else {
                arg_bail!("all arguments to be pairs [string/int key, value]", args);
            };
            let Some(key) = MapKey::try_from(key).ok() else {
                arg_bail!("all arguments to be pairs [string/int key, value]", args);
            };
            Ok((key, value.clone()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let map = HashMap::from_iter(values);
    Ok(Value::Map(Rc::new(RefCell::new(MapValue {
        inner: map,
        iteration_keys: Rc::new(RefCell::new(Vec::new())),
    }))))
}

const PUSH_ARGS: u32 = 2;
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

const SET_ARGS: u32 = 3;
pub fn builtin_set(args: &[Value]) -> ProgramFnRes {
    let [target, index_or_key, value] = args else {
        arg_bail!("list/map, int/string if map, value", args);
    };

    match target {
        Value::List(l) => {
            let index = match &index_or_key {
                Value::Integer(i) => *i as usize,
                _ => arg_bail!("list, int, value", args),
            };
            if index >= l.borrow().len() {
                out_of_bounds_bail!(l.borrow().len(), index);
            }
            l.borrow_mut()[index] = value.clone();
            fn_ok!()
        }
        Value::Map(m) => {
            let Ok(key) = MapKey::try_from(index_or_key) else {
                arg_bail!("map, int/string, value", args);
            };
            let mut mb = m.borrow_mut();
            mb.inner.insert(key, value.clone());
            mb.iteration_keys.borrow_mut().clear();
            fn_ok!()
        }
        _ => arg_bail!("list/map, int if list/string if map, value", args),
    }
}

const GET_ARGS: u32 = 2;
#[inline]
pub fn builtin_get(args: &[Value]) -> ProgramFnRes {
    let [target, index_or_key] = args else {
        arg_bail!("list/map/string, int/string if map", args);
    };

    match target {
        Value::List(l) => {
            let index = match index_or_key {
                Value::Integer(i) => *i as usize,
                _ => arg_bail!("list, int", args),
            };
            if index >= l.borrow().len() {
                out_of_bounds_bail!(l.borrow().len(), index);
            }
            Ok(l.borrow()[index].clone())
        }
        Value::Map(m) => {
            let Ok(key) = MapKey::try_from(index_or_key) else {
                arg_bail!("map, int/string", args);
            };
            match m.borrow().inner.get(&key) {
                Some(v) => Ok(v.clone()),
                None => Err(format!("Key not found in map: {}", key.dbg_display())),
            }
        }
        Value::String(s) => {
            let index = match index_or_key {
                Value::Integer(i) => *i as usize,
                _ => arg_bail!("string, int", args),
            };
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

const HAS_ARGS: u32 = 2;
pub fn builtin_has(args: &[Value]) -> ProgramFnRes {
    let [target, key] = args else {
        arg_bail!("map, string", args);
    };

    match target {
        Value::Map(m) => {
            let Ok(key_str) = MapKey::try_from(key) else {
                arg_bail!("map, string", args);
            };
            let has_key = m.borrow().inner.contains_key(&key_str);
            Ok(Value::Integer(has_key as i64))
        }
        _ => arg_bail!("map, string", args),
    }
}

const LEN_ARGS: u32 = 1;
pub fn builtin_len(args: &[Value]) -> ProgramFnRes {
    let [len] = args else {
        arg_bail!("list/string/map", args);
    };

    let len = match &len {
        Value::String(s) => s.len() as i64,
        Value::List(l) => l.borrow().len() as i64,
        Value::Map(m) => m.borrow().inner.len() as i64,
        _ => arg_bail!("list/string/map", args),
    };

    Ok(Value::Integer(len))
}

const MOD_ARGS: u32 = 2;
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

#[derive(Clone, Copy, Debug)]
pub enum ArgsRequred {
    Exact(u32),
    Range(u32, u32),
    Any,
}

impl ArgsRequred {
    pub fn matches(&self, arg_count: usize) -> bool {
        match self {
            ArgsRequred::Exact(n) => arg_count as u32 == *n,
            ArgsRequred::Range(min, max) => {
                let arg_count = arg_count as u32;
                arg_count >= *min && arg_count <= *max
            }
            ArgsRequred::Any => true,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            ArgsRequred::Exact(n) => format!("{}", n),
            ArgsRequred::Range(min, max) => format!("{} to {}", min, max),
            ArgsRequred::Any => "any number of".to_string(),
        }
    }
}

pub fn all_builtins() -> Vec<(&'static str, ProgramFn, ArgsRequred)> {
    vec![
        ("print", builtin_print, ArgsRequred::Any),
        (
            "readfile",
            builtin_readfile,
            ArgsRequred::Exact(READFILE_ARGS),
        ),
        ("split", builtin_split, ArgsRequred::Exact(SPLIT_ARGS)),
        ("int", builtin_int, ArgsRequred::Exact(INT_ARGS)),
        ("float", builtin_float, ArgsRequred::Exact(FLOAT_ARGS)),
        ("string", builtin_string, ArgsRequred::Exact(STRING_ARGS)),
        ("substr", builtin_substr, ArgsRequred::Range(2, 3)),
        ("list", builtin_list, ArgsRequred::Any),
        ("map", builtin_map, ArgsRequred::Any),
        ("push", builtin_push, ArgsRequred::Exact(PUSH_ARGS)),
        ("set", builtin_set, ArgsRequred::Exact(SET_ARGS)),
        ("get", builtin_get, ArgsRequred::Exact(GET_ARGS)),
        ("has", builtin_has, ArgsRequred::Exact(HAS_ARGS)),
        ("len", builtin_len, ArgsRequred::Exact(LEN_ARGS)),
        ("mod", builtin_mod, ArgsRequred::Exact(MOD_ARGS)),
        ("range", builtin_range, ArgsRequred::Range(1, 2)),
    ]
}
