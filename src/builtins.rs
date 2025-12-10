use std::io::{Stdout, Write};

use foldhash::HashMap;

use crate::value::{Value, ValueRef, type_display};

macro_rules! arg_bail {
    ($expected:expr, $args:expr) => {{
        return Err(format!(
            "Expects ({}), got ({})",
            $expected,
            type_display($args)
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
        Ok(Value::int(1))
    };
}

pub fn builtin_print(args: &[Value]) -> ProgramFnRes {
    let mut w = std::io::stdout();

    let print_inner = |item: &Value, w: &mut Stdout| match item.as_value_ref() {
        ValueRef::Smi(ii) => {
            write!(w, "{}", ii).unwrap();
        }
        ValueRef::Float(ff) => {
            write!(w, "{}", ff).unwrap();
        }
        ValueRef::Range(r_start, r_end) => {
            write!(w, "{}-{}", r_start, r_end).unwrap();
        }
        ValueRef::String(ss) => {
            write!(w, "{}", ss).unwrap();
        }
        ValueRef::List(_) => {
            write!(w, "<nested list>").unwrap();
        }
        ValueRef::Map(_) => {
            write!(w, "<nested map>").unwrap();
        }
    };

    for (i, arg) in args.iter().enumerate() {
        match arg.as_value_ref() {
            ValueRef::Smi(int) => {
                write!(&mut w, "{}", int).unwrap();
            }
            ValueRef::Float(f) => {
                write!(&mut w, "{}", f).unwrap();
            }
            ValueRef::Range(r_start, r_end) => {
                write!(&mut w, "{}-{}", r_start, r_end).unwrap();
            }
            ValueRef::String(s) => {
                write!(&mut w, "{}", s).unwrap();
            }
            ValueRef::List(l) => {
                write!(&mut w, "[").unwrap();
                for (j, item) in l.borrow().iter().enumerate() {
                    print_inner(item, &mut w);
                    if j < l.borrow().len() - 1 {
                        write!(&mut w, ", ").unwrap();
                    }
                }
            }
            ValueRef::Map(m) => {
                write!(&mut w, "{{").unwrap();
                let map = &m.borrow().inner;
                for (j, (key, value)) in map.iter().enumerate() {
                    write!(&mut w, "{}: ", key).unwrap();
                    print_inner(value, &mut w);
                    if j < map.len() - 1 {
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

const SLEEP_ARGS: u32 = 1;
pub fn builtin_sleep(args: &[Value]) -> ProgramFnRes {
    let [arg] = args else {
        arg_bail!("int", args);
    };

    let duration_ms = match arg.as_int() {
        Some(i) => i,
        _ => arg_bail!("int", args),
    };

    std::thread::sleep(std::time::Duration::from_millis(duration_ms as u64));
    fn_ok!()
}

const READFILE_ARGS: u32 = 1;
pub fn builtin_readfile(args: &[Value]) -> ProgramFnRes {
    let [filename] = &args else {
        arg_bail!("string", args);
    };

    let ValueRef::String(filename) = filename.as_value_ref() else {
        arg_bail!("string", args);
    };

    let content = std::fs::read_to_string(filename)
        .map_err(|_| format!("Failed to read file: {}", filename))?;

    Ok(Value::string(content.trim().into()))
}

const SPLIT_ARGS: u32 = 2;
pub fn builtin_split(args: &[Value]) -> ProgramFnRes {
    let [s, delim] = &args else {
        arg_bail!("string, string", args);
    };

    let (ValueRef::String(s), ValueRef::String(delim)) = (s.as_value_ref(), delim.as_value_ref())
    else {
        arg_bail!("string, string", args);
    };

    let parts: Vec<Value> = s
        .split(delim)
        .map(|part| Value::string(part.to_owned()))
        .collect();

    Ok(Value::list(parts))
}

const INT_ARGS: u32 = 1;
pub fn builtin_int(args: &[Value]) -> ProgramFnRes {
    let [arg] = args else {
        arg_bail!("string/float/int", args);
    };

    match arg.as_value_ref() {
        ValueRef::String(s) => {
            let int_value = s
                .parse::<i64>()
                .map_err(|_| format!("Failed to parse int from string: {}", s))?;
            Ok(Value::int(int_value))
        }
        ValueRef::Smi(i) => Ok(Value::smi(i)),
        ValueRef::Float(f) => Ok(Value::int(f as i64)),
        _ => arg_bail!("string/float/int", args),
    }
}

const FLOAT_ARGS: u32 = 1;
pub fn builtin_float(args: &[Value]) -> ProgramFnRes {
    let [arg] = args else {
        arg_bail!("string/int", args);
    };

    match arg.as_value_ref() {
        ValueRef::String(s) => {
            let val = s
                .parse::<f64>()
                .map_err(|_| format!("Failed to parse float from string: {}", s))?;
            Ok(Value::float(val))
        }
        ValueRef::Smi(i) => Ok(Value::float(i as f64)),
        ValueRef::Float(f) => Ok(Value::float(f)),
        _ => arg_bail!("string/float/int", args),
    }
}

fn write_str(w: &mut impl Write, value: &Value) {
    match value.as_value_ref() {
        ValueRef::Smi(i) => write!(w, "{}", i).unwrap(),
        ValueRef::Float(f) => write!(w, "{}", f).unwrap(),
        ValueRef::String(s) => write!(w, "{}", s).unwrap(),
        ValueRef::List(elems) => {
            for (i, v) in elems.borrow().iter().enumerate() {
                if i > 0 {
                    write!(w, ",").unwrap();
                }
                write_str(w, v);
            }
        }
        ValueRef::Map(map) => {
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
        ValueRef::Range(s, e) => write!(w, "{}-{}", s, e).unwrap(),
    }
}

const STRING_ARGS: u32 = 1;
pub fn builtin_string(args: &[Value]) -> ProgramFnRes {
    let [arg] = args else {
        arg_bail!("value", args);
    };
    let mut w = Vec::new();
    write_str(&mut w, arg);

    // SAFETY: I'm only writing valid UTF-8 data to the vector.
    let s = unsafe { String::from_utf8(w).unwrap_unchecked() };

    Ok(Value::string(s))
}

pub fn builtin_substr(args: &[Value]) -> ProgramFnRes {
    if args.len() < 2 || args.len() > 3 {
        arg_bail!("string, int, opt int", args);
    }
    let string = match args[0].as_value_ref() {
        ValueRef::String(s) => s,
        _ => arg_bail!("string, int, opt int", args),
    };
    let start = match args[1].as_int() {
        Some(i) => i,
        _ => arg_bail!("string, int, opt int", args),
    };
    let mut end = if args.get(2).is_some() {
        match args[2].as_int() {
            Some(i) => i,
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
    Ok(Value::string(substring.to_owned()))
}

pub fn builtin_list(args: &[Value]) -> ProgramFnRes {
    let list = args.to_vec();
    Ok(Value::list(list))
}

pub fn builtin_map(args: &[Value]) -> ProgramFnRes {
    let values = args
        .iter()
        .map(|arg| {
            let ValueRef::List(pair) = arg.as_value_ref() else {
                arg_bail!("all arguments to be pairs [string key, value]", args);
            };
            let [key, value] = &pair.borrow()[..] else {
                arg_bail!("all arguments to be pairs [string key, value]", args);
            };
            if !key.is_string() {
                arg_bail!("all arguments to be pairs [string key, value]", args);
            };
            Ok((key.clone(), value.clone()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let map = HashMap::from_iter(values);
    Ok(Value::map(map))
}

const PUSH_ARGS: u32 = 2;
#[inline]
pub fn builtin_push(args: &[Value]) -> ProgramFnRes {
    let [target, value] = args else {
        arg_bail!("list, value", args);
    };

    match target.as_value_ref() {
        ValueRef::List(l) => {
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

    match target.as_value_ref() {
        ValueRef::List(l) => {
            let index = match index_or_key.as_int() {
                Some(i) => i as usize,
                None => arg_bail!("list, int, value", args),
            };
            if index >= l.borrow().len() {
                out_of_bounds_bail!(l.borrow().len(), index);
            }
            l.borrow_mut()[index] = value.clone();
            fn_ok!()
        }
        ValueRef::Map(m) => {
            if !index_or_key.is_string() {
                arg_bail!("map, string, value", args);
            };
            let mut mb = m.borrow_mut();
            mb.inner.insert(index_or_key.clone(), value.clone());
            mb.iter_keys.clear();
            fn_ok!()
        }
        _ => arg_bail!("list/map, int if list/string if map, value", args),
    }
}

const GET_ARGS: u32 = 2;
#[inline]
pub fn builtin_get(args: &[Value]) -> ProgramFnRes {
    let [target, index_or_key] = args else {
        arg_bail!("list/string/range/map, int", args);
    };

    match target.as_value_ref() {
        ValueRef::Range(start, end) => {
            let Some(index) = index_or_key.as_int() else {
                arg_bail!("list/string/range, int", args);
            };

            if index >= end - start {
                out_of_bounds_bail!(end - start, index);
            }
            Ok(Value::int(start + index))
        }
        ValueRef::String(s) => {
            let Some(index) = index_or_key.as_int() else {
                arg_bail!("list/string/range, int", args);
            };

            let index = index as usize;

            if index >= s.len() {
                out_of_bounds_bail!(s.len(), index);
            }
            Ok(Value::string(s.chars().nth(index).unwrap().to_string()))
        }
        ValueRef::List(l) => {
            let Some(index) = index_or_key.as_int() else {
                arg_bail!("list/string/range, int", args);
            };

            let index = index as usize;

            let l = l.borrow();

            if index >= l.len() {
                out_of_bounds_bail!(l.len(), index);
            }
            Ok(l[index].clone())
        }
        ValueRef::Map(m) => {
            if !index_or_key.is_string() {
                arg_bail!("map, string", args);
            };

            let m = &m.borrow().inner;

            match m.get(index_or_key) {
                Some(v) => Ok(v.clone()),
                None => Err(format!("Key not found in map: {}", index_or_key)),
            }
        }
        _ => {
            arg_bail!("list/map/string, int/string if map", args);
        }
    }
}

const HAS_ARGS: u32 = 2;
pub fn builtin_has(args: &[Value]) -> ProgramFnRes {
    let [target, key] = args else {
        arg_bail!("map, string", args);
    };

    match target.as_value_ref() {
        ValueRef::Map(m) => {
            if !key.is_string() {
                arg_bail!("map, string", args);
            };
            let has_key = m.borrow().inner.contains_key(key);
            Ok(Value::bool(has_key))
        }
        _ => arg_bail!("map, string", args),
    }
}

const LEN_ARGS: u32 = 1;
pub fn builtin_len(args: &[Value]) -> ProgramFnRes {
    let [len] = args else {
        arg_bail!("list/string/map", args);
    };

    let len = match len.as_value_ref() {
        ValueRef::String(s) => s.len() as i64,
        ValueRef::List(l) => l.borrow().len() as i64,
        ValueRef::Map(m) => m.borrow().inner.len() as i64,
        _ => arg_bail!("list/string/map", args),
    };

    Ok(Value::int(len))
}

const MOD_ARGS: u32 = 2;
pub fn builtin_mod(args: &[Value]) -> ProgramFnRes {
    let [a, b] = args else {
        arg_bail!("int, int", args);
    };

    let Some(a) = a.as_int() else {
        arg_bail!("int, int", args);
    };

    let Some(b) = b.as_int() else {
        arg_bail!("int, int", args);
    };

    if b == 0 {
        return Err("Division by zero in mod operation".to_string());
    }

    Ok(Value::int(a % b))
}

pub fn builtin_range(args: &[Value]) -> ProgramFnRes {
    if args.is_empty() || args.len() > 2 {
        arg_bail!("int, opt int", args);
    };

    let start = &args[0];

    let Some(mut start) = start.as_int() else {
        arg_bail!("int, opt int", args);
    };

    let end = if args.len() == 2 {
        let end_arg = &args[1];
        let Some(end) = end_arg.as_int() else {
            arg_bail!("int, opt int", args);
        };
        end
    } else {
        let tmp = start;
        start = 0;
        tmp
    };

    Ok(Value::range(start, end))
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
        ("sleep", builtin_sleep, ArgsRequred::Exact(SLEEP_ARGS)),
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
