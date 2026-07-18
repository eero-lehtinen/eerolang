use std::{
    collections::VecDeque,
    io::{Stdout, Write},
};

use foldhash::HashMap;

use crate::{
    EXTRA_ARGS,
    value::{Value, ValueRef, type_display},
};

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
        Ok(Value::null())
    };
}

pub fn builtin_list(args: &[Value]) -> ProgramFnRes {
    let list = args.to_vec();
    Ok(Value::list(list))
}

pub fn builtin_queue(args: &[Value]) -> ProgramFnRes {
    let queue = VecDeque::from(args.to_vec());
    Ok(Value::queue(queue))
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

const ARGS_ARGS: u32 = 0;
pub fn builtin_args(_: &[Value]) -> ProgramFnRes {
    Ok(Value::list(
        EXTRA_ARGS
            .get()
            .map(|args| args.iter().map(|s| Value::string(s.clone())).collect())
            .unwrap_or_default(),
    ))
}

const NOT_ARGS: u32 = 1;
pub fn builtin_not(args: &[Value]) -> ProgramFnRes {
    let [arg] = args else {
        arg_bail!("any", args);
    };

    Ok(Value::smi(if arg.is_falsy() { 1 } else { 0 }))
}

fn print_inner(item: &Value, depth: u32, w: &mut Stdout) {
    match item.as_value_ref() {
        ValueRef::Null => {
            write!(w, "null").unwrap();
        }
        ValueRef::Smi(int) => {
            write!(w, "{}", int).unwrap();
        }
        ValueRef::Float(f) => {
            write!(w, "{}", f).unwrap();
        }
        ValueRef::Range(r) => {
            write!(w, "{}-{}", r.start, r.end).unwrap();
        }
        ValueRef::String(s) => {
            if depth == 0 {
                write!(w, "{}", s).unwrap();
            } else {
                write!(w, "\"{}\"", s).unwrap();
            }
        }
        ValueRef::List(l) => {
            if depth > 2 {
                write!(w, "[...]").unwrap();
                return;
            }
            write!(w, "[").unwrap();
            for (j, item) in l.borrow().iter().enumerate() {
                print_inner(item, depth + 1, w);
                if j < l.borrow().len() - 1 {
                    write!(w, ", ").unwrap();
                }
            }
            write!(w, "]").unwrap();
        }
        ValueRef::Queue(q) => {
            if depth > 2 {
                write!(w, "queue[...]").unwrap();
                return;
            }
            write!(w, "queue[").unwrap();
            for (j, item) in q.borrow().iter().enumerate() {
                print_inner(item, depth + 1, w);
                if j < q.borrow().len() - 1 {
                    write!(w, ", ").unwrap();
                }
            }
            write!(w, "]").unwrap();
        }
        ValueRef::Map(m) => {
            if depth > 2 {
                write!(w, "{{...}}").unwrap();
                return;
            }
            write!(w, "{{").unwrap();
            let map = &m.borrow().inner;
            for (j, (key, value)) in map.iter().enumerate() {
                let key = match key.as_value_ref() {
                    ValueRef::String(s) => s,
                    _ => unreachable!(),
                };
                write!(w, "{}: ", key).unwrap();
                print_inner(value, depth + 1, w);
                if j < map.len() - 1 {
                    write!(w, ", ").unwrap();
                }
            }
            write!(w, "}}").unwrap();
        }
        ValueRef::Function(ip) => {
            write!(w, "<fn @{}>", ip).unwrap();
        }
        ValueRef::Builtin(idx) => {
            write!(w, "<builtin #{}>", idx).unwrap();
        }
    };
}

pub fn builtin_print(args: &[Value]) -> ProgramFnRes {
    let mut w = std::io::stdout();

    for (i, arg) in args.iter().enumerate() {
        print_inner(arg, 0, &mut w);
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

    Ok(Value::string(content))
}

const READBYTES_ARGS: u32 = 1;
pub fn builtin_readbytes(args: &[Value]) -> ProgramFnRes {
    let [filename] = args else {
        arg_bail!("string", args);
    };
    let ValueRef::String(filename) = filename.as_value_ref() else {
        arg_bail!("string", args);
    };

    let bytes = std::fs::read(filename)
        .map_err(|error| format!("Failed to read file '{}': {}", filename, error))?;
    Ok(Value::list(
        bytes
            .into_iter()
            .map(|byte| Value::int(byte.into()))
            .collect(),
    ))
}

const TRIM_ARGS: u32 = 1;
pub fn builtin_trim(args: &[Value]) -> ProgramFnRes {
    let [s] = &args else {
        arg_bail!("string", args);
    };

    let ValueRef::String(s) = s.as_value_ref() else {
        arg_bail!("string", args);
    };

    let trimmed = s.trim();

    Ok(Value::string(trimmed.to_owned()))
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
        ValueRef::Null => write!(w, "null").unwrap(),
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
        ValueRef::Queue(q) => {
            let q_ref = q.borrow();
            for (i, v) in q_ref.iter().enumerate() {
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
        ValueRef::Range(r) => write!(w, "{}-{}", r.start, r.end).unwrap(),
        ValueRef::Function(ip) => write!(w, "<fn @{}>", ip).unwrap(),
        ValueRef::Builtin(idx) => write!(w, "<builtin #{}>", idx).unwrap(),
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
    let char_count = string.chars().count() as i64;
    let mut end = if args.get(2).is_some() {
        match args[2].as_int() {
            Some(i) => i,
            _ => arg_bail!("string, int, opt int", args),
        }
    } else {
        char_count
    };

    if start < 0 || start > end {
        return Err(format!(
            "Expects start to be non-negative and less than end, got start: {}, end: {}",
            start, end
        ));
    }
    if end < 0 {
        end = char_count - end.abs();
    }
    end = end.min(char_count);

    let substring: String = string
        .chars()
        .skip(start as usize)
        .take((end - start) as usize)
        .collect();
    Ok(Value::string(substring))
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
        ValueRef::Queue(q) => {
            q.borrow_mut().push_back(value.clone());
            fn_ok!()
        }
        _ => arg_bail!("list, value", args),
    }
}

const POP_ARGS: u32 = 1;
pub fn builtin_pop(args: &[Value]) -> ProgramFnRes {
    let [target] = args else {
        arg_bail!("list/queue", args);
    };

    match target.as_value_ref() {
        ValueRef::List(l) => {
            let mut lb = l.borrow_mut();
            if lb.is_empty() {
                return Err("Cannot pop from an empty list".to_string());
            }
            let value = lb.pop().unwrap();
            Ok(value)
        }
        ValueRef::Queue(q) => {
            let mut qb = q.borrow_mut();
            if qb.is_empty() {
                return Err("Cannot pop from an empty queue".to_string());
            }
            let value = qb.pop_back().unwrap();
            Ok(value)
        }
        _ => arg_bail!("list/queue", args),
    }
}

const POP_FRONT_ARGS: u32 = 1;
pub fn builtin_pop_front(args: &[Value]) -> ProgramFnRes {
    let [target] = args else {
        arg_bail!("list/queue", args);
    };

    match target.as_value_ref() {
        ValueRef::List(l) => {
            let mut lb = l.borrow_mut();
            if lb.is_empty() {
                return Err("Cannot pop from an empty list".to_string());
            }
            let value = lb.remove(0);
            Ok(value)
        }
        ValueRef::Queue(q) => {
            let mut qb = q.borrow_mut();
            if qb.is_empty() {
                return Err("Cannot pop from an empty queue".to_string());
            }
            let value = qb.pop_front().unwrap();
            Ok(value)
        }
        _ => arg_bail!("list/queue", args),
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
            let mut l = l.borrow_mut();
            if index >= l.len() {
                out_of_bounds_bail!(l.len(), index);
            }
            l[index] = value.clone();
            fn_ok!()
        }
        ValueRef::Queue(q) => {
            let index = match index_or_key.as_int() {
                Some(i) => i as usize,
                None => arg_bail!("queue, int, value", args),
            };
            let mut qb = q.borrow_mut();
            if index >= qb.len() {
                out_of_bounds_bail!(qb.len(), index);
            }
            qb[index] = value.clone();
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
        arg_bail!("list/queue/string/range/map, int", args);
    };

    match target.as_value_ref() {
        ValueRef::Range(r) => {
            let Some(index) = index_or_key.as_int() else {
                arg_bail!("list/string/range, int", args);
            };
            let dist = (r.end - r.start).abs();
            if index >= dist {
                out_of_bounds_bail!(dist, index);
            }
            Ok(Value::int(index))
        }
        ValueRef::String(s) => {
            let Some(index) = index_or_key.as_int() else {
                arg_bail!("list/string/range, int", args);
            };

            let index = index as usize;

            match s.chars().nth(index) {
                Some(c) => Ok(Value::string(c.to_string())),
                None => out_of_bounds_bail!(s.chars().count(), index),
            }
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
        ValueRef::Queue(q) => {
            let Some(index) = index_or_key.as_int() else {
                arg_bail!("list/string/range, int", args);
            };

            let index = index as usize;

            let q = q.borrow();

            if index >= q.len() {
                out_of_bounds_bail!(q.len(), index);
            }
            Ok(q[index].clone())
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

const REMOVE_ARGS: u32 = 2;
pub fn builtin_remove(args: &[Value]) -> ProgramFnRes {
    let [target, key] = args else {
        arg_bail!("map/list/queue, string/int", args);
    };

    match target.as_value_ref() {
        ValueRef::Map(m) => {
            if !key.is_string() {
                arg_bail!("map, string", args);
            };
            let mut mb = m.borrow_mut();
            let removed = mb.inner.remove(key).is_some();
            mb.iter_keys.clear();
            Ok(Value::bool(removed))
        }
        ValueRef::List(l) => {
            let Some(index) = key.as_int() else {
                arg_bail!("list, int", args);
            };
            let index = index as usize;
            let mut l = l.borrow_mut();
            if index >= l.len() {
                out_of_bounds_bail!(l.len(), index);
            }
            l.remove(index);
            fn_ok!()
        }
        ValueRef::Queue(q) => {
            let Some(index) = key.as_int() else {
                arg_bail!("queue, int", args);
            };
            let index = index as usize;
            let mut q = q.borrow_mut();
            if index >= q.len() {
                out_of_bounds_bail!(q.len(), index);
            }
            q.remove(index);
            fn_ok!()
        }
        _ => arg_bail!("map, string", args),
    }
}

const CLONE_ARGS: u32 = 1;
pub fn builtin_clone(args: &[Value]) -> ProgramFnRes {
    let [target] = args else {
        arg_bail!("any", args);
    };

    Ok(clone_impl(target))
}

fn clone_impl(value: &Value) -> Value {
    match value.as_value_ref() {
        ValueRef::List(l) => {
            let cloned_list = l.borrow().iter().map(clone_impl).collect();
            Value::list(cloned_list)
        }
        ValueRef::Queue(q) => {
            let cloned_queue = q.borrow().iter().map(clone_impl).collect();
            Value::queue(cloned_queue)
        }
        ValueRef::Map(m) => {
            let cloned_map = m
                .borrow()
                .inner
                .iter()
                .map(|(k, v)| (k.clone(), clone_impl(v)))
                .collect();
            Value::map(cloned_map)
        }
        _ => value.clone(),
    }
}

const LEN_ARGS: u32 = 1;
pub fn builtin_len(args: &[Value]) -> ProgramFnRes {
    let [len] = args else {
        arg_bail!("list/string/map", args);
    };

    let len = match len.as_value_ref() {
        ValueRef::String(s) => s.chars().count() as i64,
        ValueRef::List(l) => l.borrow().len() as i64,
        ValueRef::Queue(q) => q.borrow().len() as i64,
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

fn number_pair(args: &[Value]) -> Result<(&Value, &Value), String> {
    let [left, right] = args else {
        return Err(format!(
            "Expects (number, number), got ({})",
            type_display(args)
        ));
    };
    if left.as_number().is_none() || right.as_number().is_none() {
        return Err(format!(
            "Expects (number, number), got ({})",
            type_display(args)
        ));
    };
    Ok((left, right))
}

fn to_uint32(value: &Value) -> u32 {
    let number = value.as_number().unwrap();
    if !number.is_finite() || number == 0.0 {
        return 0;
    }
    number.trunc().rem_euclid(4_294_967_296.0) as u32
}

fn to_int32(value: &Value) -> i32 {
    to_uint32(value) as i32
}

pub fn builtin_bit_and(args: &[Value]) -> ProgramFnRes {
    let (left, right) = number_pair(args)?;
    Ok(Value::smi(to_int32(left) & to_int32(right)))
}

pub fn builtin_bit_or(args: &[Value]) -> ProgramFnRes {
    let (left, right) = number_pair(args)?;
    Ok(Value::smi(to_int32(left) | to_int32(right)))
}

pub fn builtin_bit_xor(args: &[Value]) -> ProgramFnRes {
    let (left, right) = number_pair(args)?;
    Ok(Value::smi(to_int32(left) ^ to_int32(right)))
}

const BIT_NOT_ARGS: u32 = 1;
pub fn builtin_bit_not(args: &[Value]) -> ProgramFnRes {
    let [value] = args else {
        arg_bail!("number", args);
    };
    if value.as_number().is_none() {
        arg_bail!("number", args);
    };
    Ok(Value::smi(!to_int32(value)))
}

fn shift_args(args: &[Value]) -> Result<(i32, u32), String> {
    let (value, amount) = number_pair(args)?;
    Ok((to_int32(value), to_uint32(amount) & 31))
}

pub fn builtin_shift_left(args: &[Value]) -> ProgramFnRes {
    let (value, amount) = shift_args(args)?;
    Ok(Value::smi(value.wrapping_shl(amount)))
}

pub fn builtin_shift_right(args: &[Value]) -> ProgramFnRes {
    let (value, amount) = shift_args(args)?;
    Ok(Value::smi(value >> amount))
}

pub fn builtin_shift_right_unsigned(args: &[Value]) -> ProgramFnRes {
    let (value, amount) = shift_args(args)?;
    Ok(Value::int(((value as u32) >> amount).into()))
}

const POW_ARGS: u32 = 2;
pub fn builtin_pow(args: &[Value]) -> ProgramFnRes {
    let [base, exponent] = args else {
        arg_bail!("int/float, int/float", args);
    };

    let base_f = match base.as_value_ref() {
        ValueRef::Smi(i) => i as f64,
        ValueRef::Float(f) => f,
        _ => arg_bail!("int/float, int/float", args),
    };

    let exponent_f = match exponent.as_value_ref() {
        ValueRef::Smi(i) => i as f64,
        ValueRef::Float(f) => f,
        _ => arg_bail!("int/float, int/float", args),
    };

    let result = base_f.powf(exponent_f);
    Ok(Value::number(result))
}

const SQRT_ARGS: u32 = 1;
pub fn builtin_sqrt(args: &[Value]) -> ProgramFnRes {
    let [arg] = args else {
        arg_bail!("int/float", args);
    };

    match arg.as_value_ref() {
        ValueRef::Smi(i) => {
            if i < 0 {
                return Err("Cannot compute square root of a negative integer".to_string());
            }
            let result = (i as f64).sqrt();
            Ok(Value::number(result))
        }
        ValueRef::Float(f) => {
            if f < 0.0 {
                return Err("Cannot compute square root of a negative float".to_string());
            }
            let result = f.sqrt();
            Ok(Value::number(result))
        }
        _ => arg_bail!("int/float", args),
    }
}

pub fn builtin_min(args: &[Value]) -> ProgramFnRes {
    if args.len() < 2 {
        arg_bail!("at least 2 int/float", args);
    }

    let Some(mut min) = args[0].as_number() else {
        arg_bail!("at least 2 int/float", args);
    };

    for arg in &args[1..] {
        let Some(num) = arg.as_number() else {
            arg_bail!("at least 2 int/float", args);
        };
        if num < min {
            min = num;
        }
    }

    Ok(Value::number(min))
}

pub fn builtin_max(args: &[Value]) -> ProgramFnRes {
    if args.len() < 2 {
        arg_bail!("at least 2 int/float", args);
    }

    let Some(mut max) = args[0].as_number() else {
        arg_bail!("at least 2 int/float", args);
    };

    for arg in &args[1..] {
        let Some(num) = arg.as_number() else {
            arg_bail!("at least 2 int/float", args);
        };
        if num > max {
            max = num;
        }
    }

    Ok(Value::number(max))
}

pub type ProgramFnRes = Result<Value, String>;
pub type ProgramFn = fn(&[Value]) -> ProgramFnRes;

#[derive(Clone, Copy, Debug)]
pub enum ArgsRequred {
    Exact(u32),
    Range(u32, u32),
    AtLeast(u32),
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
            ArgsRequred::AtLeast(n) => arg_count as u32 >= *n,
            ArgsRequred::Any => true,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            ArgsRequred::Exact(n) => format!("{}", n),
            ArgsRequred::Range(min, max) => format!("{} to {}", min, max),
            ArgsRequred::AtLeast(n) => format!("at least {}", n),
            ArgsRequred::Any => "any number of".to_string(),
        }
    }
}

pub fn all_builtins() -> Vec<(&'static str, ProgramFn, ArgsRequred)> {
    vec![
        ("list", builtin_list, ArgsRequred::Any),
        ("queue", builtin_queue, ArgsRequred::Any),
        ("map", builtin_map, ArgsRequred::Any),
        ("range", builtin_range, ArgsRequred::Range(1, 2)),
        ("args", builtin_args, ArgsRequred::Exact(ARGS_ARGS)),
        ("not", builtin_not, ArgsRequred::Exact(NOT_ARGS)),
        ("print", builtin_print, ArgsRequred::Any),
        ("sleep", builtin_sleep, ArgsRequred::Exact(SLEEP_ARGS)),
        (
            "readfile",
            builtin_readfile,
            ArgsRequred::Exact(READFILE_ARGS),
        ),
        (
            "readbytes",
            builtin_readbytes,
            ArgsRequred::Exact(READBYTES_ARGS),
        ),
        ("trim", builtin_trim, ArgsRequred::Exact(TRIM_ARGS)),
        ("split", builtin_split, ArgsRequred::Exact(SPLIT_ARGS)),
        ("int", builtin_int, ArgsRequred::Exact(INT_ARGS)),
        ("float", builtin_float, ArgsRequred::Exact(FLOAT_ARGS)),
        ("string", builtin_string, ArgsRequred::Exact(STRING_ARGS)),
        ("substr", builtin_substr, ArgsRequred::Range(2, 3)),
        ("push", builtin_push, ArgsRequred::Exact(PUSH_ARGS)),
        ("pop", builtin_pop, ArgsRequred::Exact(POP_ARGS)),
        (
            "pop_front",
            builtin_pop_front,
            ArgsRequred::Exact(POP_FRONT_ARGS),
        ),
        ("set", builtin_set, ArgsRequred::Exact(SET_ARGS)),
        ("get", builtin_get, ArgsRequred::Exact(GET_ARGS)),
        ("has", builtin_has, ArgsRequred::Exact(HAS_ARGS)),
        ("remove", builtin_remove, ArgsRequred::Exact(REMOVE_ARGS)),
        ("clone", builtin_clone, ArgsRequred::Exact(CLONE_ARGS)),
        ("len", builtin_len, ArgsRequred::Exact(LEN_ARGS)),
        ("mod", builtin_mod, ArgsRequred::Exact(MOD_ARGS)),
        ("bit_and", builtin_bit_and, ArgsRequred::Exact(2)),
        ("bit_or", builtin_bit_or, ArgsRequred::Exact(2)),
        ("bit_xor", builtin_bit_xor, ArgsRequred::Exact(2)),
        ("bit_not", builtin_bit_not, ArgsRequred::Exact(BIT_NOT_ARGS)),
        ("shift_left", builtin_shift_left, ArgsRequred::Exact(2)),
        ("shift_right", builtin_shift_right, ArgsRequred::Exact(2)),
        (
            "shift_right_unsigned",
            builtin_shift_right_unsigned,
            ArgsRequred::Exact(2),
        ),
        ("pow", builtin_pow, ArgsRequred::Exact(POW_ARGS)),
        ("sqrt", builtin_sqrt, ArgsRequred::Exact(SQRT_ARGS)),
        ("min", builtin_min, ArgsRequred::AtLeast(2)),
        ("max", builtin_max, ArgsRequred::AtLeast(2)),
    ]
}
