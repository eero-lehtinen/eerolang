use std::cell::RefCell;
use std::marker::PhantomData;
use std::mem;
use std::rc::Rc;

use std::hash::Hash;

use foldhash::HashMap;

/// A tagged pointer that holds either an Rc<T> or a small integer.
///
/// We use the Least Significant Bit (LSB) as the tag:
/// 0 = Pointer
/// 1 = Integer
pub struct Value {
    bits: usize,
    _marker: PhantomData<Rc<ValueInner>>,
}

pub enum IntOrRc<'a> {
    Int(i32),
    Rc(&'a ValueInner),
}

pub enum ValueRef<'a> {
    Smi(i32),
    Float(f64),
    Range(i64, i64),
    String(&'a str),
    List(&'a RefCell<Vec<Value>>),
    Map(&'a RefCell<Map>),
}

pub enum ValueInner {
    Float(f64),
    Range(i64, i64),
    String(String),
    List(RefCell<Vec<Value>>),
    Map(RefCell<Map>),
}

pub struct Map {
    pub inner: HashMap<Value, Value>,
    pub iter_keys: Vec<Value>,
}

pub enum OpError {
    DivisionByZero,
    InvalidOperandTypes,
}

pub type OpResult = Result<Value, OpError>;

impl Value {
    const TAG_MASK: usize = 0b1;
    const INT_FLAG: usize = 0b1;

    pub fn smi(val: i32) -> Self {
        let val_usize = val as usize;

        let bits = (val_usize << 32) | Self::INT_FLAG;

        Self {
            bits,
            _marker: PhantomData,
        }
    }

    fn rc(rc: Rc<ValueInner>) -> Self {
        assert!(
            mem::align_of::<ValueInner>() > 1,
            "Type T must have alignment > 1"
        );

        let ptr = Rc::into_raw(rc);
        let bits = ptr as usize;

        assert!(
            bits & Self::TAG_MASK == 0,
            "Pointer was not properly aligned"
        );

        Self {
            bits,
            _marker: PhantomData,
        }
    }

    pub fn float(val: f64) -> Self {
        Self::rc(Rc::new(ValueInner::Float(val)))
    }

    pub fn int(val: i64) -> Self {
        if val < i32::MIN as i64 || val > i32::MAX as i64 {
            Self::float(val as f64)
        } else {
            Self::smi(val as i32)
        }
    }

    pub fn range(start: i64, end: i64) -> Self {
        Self::rc(Rc::new(ValueInner::Range(start, end)))
    }

    pub fn string(val: String) -> Self {
        Self::rc(Rc::new(ValueInner::String(val)))
    }

    pub fn list(val: Vec<Value>) -> Self {
        Self::rc(Rc::new(ValueInner::List(RefCell::new(val))))
    }

    pub fn bool(val: bool) -> Self {
        Self::smi(if val { 1 } else { 0 })
    }

    pub fn map(val: HashMap<Value, Value>) -> Self {
        Self::rc(Rc::new(ValueInner::Map(RefCell::new(Map {
            inner: val,
            iter_keys: Vec::new(),
        }))))
    }

    pub fn is_smi(&self) -> bool {
        (self.bits & Self::TAG_MASK) == Self::INT_FLAG
    }

    pub fn is_rc(&self) -> bool {
        !self.is_smi()
    }

    pub fn as_int(&self) -> Option<i64> {
        match self.as_value_ref() {
            ValueRef::Smi(i) => Some(i as i64),
            ValueRef::Float(f) => (f == f.floor()).then_some(f as i64),
            _ => None,
        }
    }

    pub fn as_value_ref(&self) -> ValueRef<'_> {
        if self.is_smi() {
            ValueRef::Smi((self.bits >> 32) as i32)
        } else {
            let ptr = self.bits as *const ValueInner;
            unsafe {
                match &*ptr {
                    ValueInner::Float(f) => ValueRef::Float(*f),
                    ValueInner::Range(s, e) => ValueRef::Range(*s, *e),
                    ValueInner::String(s) => ValueRef::String(s),
                    ValueInner::List(lst) => ValueRef::List(lst),
                    ValueInner::Map(map) => ValueRef::Map(map),
                }
            }
        }
    }

    pub fn is_string(&self) -> bool {
        matches!(self.as_value_ref(), ValueRef::String(_))
    }

    pub fn as_int_or_rc(&'_ self) -> IntOrRc<'_> {
        if self.is_smi() {
            IntOrRc::Int((self.bits >> 32) as i32)
        } else {
            let ptr = self.bits as *const ValueInner;
            unsafe { IntOrRc::Rc(&*ptr) }
        }
    }

    pub fn add(&self, other: &Self) -> OpResult {
        let res = match (self.as_value_ref(), other.as_value_ref()) {
            (ValueRef::Smi(a), ValueRef::Smi(b)) => {
                if let Some(result) = a.checked_add(b) {
                    Value::smi(result)
                } else {
                    Value::float(a as f64 + b as f64)
                }
            }
            (ValueRef::Smi(a), ValueRef::Float(b)) => Value::float(a as f64 + b),
            (ValueRef::Float(a), ValueRef::Smi(b)) => Value::float(a + b as f64),
            (ValueRef::Float(a), ValueRef::Float(b)) => Value::float(a + b),
            (ValueRef::String(a), ValueRef::String(b)) => {
                let mut s = String::with_capacity(a.len() + b.len());
                s.push_str(a);
                s.push_str(b);
                Value::string(s)
            }
            _ => return Err(OpError::InvalidOperandTypes),
        };
        Ok(res)
    }

    fn eq_impl(&self, other: &Self) -> Option<bool> {
        let res = match (self.as_value_ref(), other.as_value_ref()) {
            (ValueRef::Smi(a), ValueRef::Smi(b)) => a == b,
            (ValueRef::Smi(a), ValueRef::Float(b)) => a as f64 == b,
            (ValueRef::Float(a), ValueRef::Smi(b)) => a == b as f64,
            (ValueRef::Float(a), ValueRef::Float(b)) => a == b,
            (ValueRef::Range(s1, e1), ValueRef::Range(s2, e2)) => s1 == s2 && e1 == e2,
            (ValueRef::String(a), ValueRef::String(b)) => a == b,
            _ => return None,
        };
        Some(res)
    }

    pub fn eq(&self, other: &Self) -> OpResult {
        if let Some(r) = self.eq_impl(other) {
            Ok(Value::smi(if r { 1 } else { 0 }))
        } else {
            Err(OpError::InvalidOperandTypes)
        }
    }

    pub fn neq(&self, other: &Self) -> OpResult {
        if let Some(r) = self.eq_impl(other) {
            Ok(Value::smi(if !r { 1 } else { 0 }))
        } else {
            Err(OpError::InvalidOperandTypes)
        }
    }

    pub fn div(&self, other: &Self) -> OpResult {
        let res = match (self.as_value_ref(), other.as_value_ref()) {
            (ValueRef::Smi(a), ValueRef::Smi(b)) => {
                if b == 0 {
                    return Err(OpError::DivisionByZero);
                }
                if let Some(result) = a.checked_div(b) {
                    Value::smi(result)
                } else {
                    Value::float(a as f64 / b as f64)
                }
            }
            (ValueRef::Smi(a), ValueRef::Float(b)) => {
                if b == 0.0 {
                    return Err(OpError::DivisionByZero);
                }
                Value::float(a as f64 / b)
            }
            (ValueRef::Float(a), ValueRef::Smi(b)) => {
                if b == 0 {
                    return Err(OpError::DivisionByZero);
                }
                Value::float(a / b as f64)
            }
            (ValueRef::Float(a), ValueRef::Float(b)) => {
                if b == 0.0 {
                    return Err(OpError::DivisionByZero);
                }
                Value::float(a / b)
            }
            _ => return Err(OpError::InvalidOperandTypes),
        };
        Ok(res)
    }

    pub fn dbg_display(&self) -> String {
        format!("{:?}", self)
    }
}

macro_rules! op_impl {
    ($func_name:ident, $checked_op:ident, $op:tt) => {
        impl Value {
            pub fn $func_name(&self, other: &Self) -> OpResult {
                let res = match (self.as_value_ref(), other.as_value_ref()) {
                    (ValueRef::Smi(a), ValueRef::Smi(b)) => {
                        if let Some(result) = a.$checked_op(b) {
                            Value::smi(result)
                        } else {
                            Value::float(a as f64 $op b as f64)
                        }
                    }
                    (ValueRef::Smi(a), ValueRef::Float(b)) => Value::float(a as f64 $op b),
                    (ValueRef::Float(a), ValueRef::Smi(b)) => Value::float(a $op b as f64),
                    (ValueRef::Float(a), ValueRef::Float(b)) => Value::float(a $op b),
                    _ => return Err(OpError::InvalidOperandTypes),
                };
                Ok(res)
            }
        }
    };
}

op_impl!(sub, checked_sub, -);
op_impl!(mul, checked_mul, *);

macro_rules! cmp_op_impl {
    ($func_name:ident, $cmp_op:tt) => {
        impl Value {
            pub fn $func_name(&self, other: &Self) -> OpResult {
                let res = match (self.as_value_ref(), other.as_value_ref()) {
                    (ValueRef::Smi(a), ValueRef::Smi(b)) => a $cmp_op b,
                    (ValueRef::Smi(a), ValueRef::Float(b)) => (a as f64) $cmp_op b,
                    (ValueRef::Float(a), ValueRef::Smi(b)) => a $cmp_op b as f64,
                    (ValueRef::Float(a), ValueRef::Float(b)) => a $cmp_op b,
                    _ => return Err(OpError::InvalidOperandTypes),
                };
                Ok(Value::smi(if res { 1 } else { 0 }))
            }
        }
    };
}

cmp_op_impl!(lt, <);
cmp_op_impl!(gt, >);
cmp_op_impl!(lte,<=);
cmp_op_impl!(gte,>=);

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.eq_impl(other).unwrap_or_default()
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self.as_value_ref() {
            ValueRef::String(s) => {
                s.hash(state);
            }
            _ => {
                panic!("Cannot hash non-string values");
            }
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Self::smi(0)
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        if self.is_rc() {
            let ptr = self.bits as *const ValueInner;
            unsafe {
                let _ = Rc::from_raw(ptr);
            }
        }
    }
}

impl Clone for Value {
    fn clone(&self) -> Self {
        if self.is_rc() {
            let ptr = self.bits as *const ValueInner;
            unsafe {
                Rc::increment_strong_count(ptr);
            }
            Self {
                bits: self.bits,
                _marker: PhantomData,
            }
        } else {
            Self {
                bits: self.bits,
                _marker: PhantomData,
            }
        }
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_int_or_rc() {
            IntOrRc::Int(i) => write!(f, "Int({})", i),
            IntOrRc::Rc(rc) => match rc {
                ValueInner::Float(fl) => write!(f, "Float({})", fl),
                ValueInner::Range(start, end) => write!(f, "Range({}, {})", start, end),
                ValueInner::String(s) => write!(f, "String({})", s),
                ValueInner::List(lst) => {
                    write!(f, "List([")?;
                    for (i, val) in lst.borrow().iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:?}", val)?;
                    }
                    write!(f, "])")
                }
                ValueInner::Map(map) => {
                    write!(f, "Map{{")?;
                    for (i, (key, val)) in map.borrow().inner.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:?}: {:?}", key, val)?;
                    }
                    write!(f, "}}")
                }
            },
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_int_or_rc() {
            IntOrRc::Int(i) => write!(f, "num {}", i),
            IntOrRc::Rc(rc) => match rc {
                ValueInner::Float(fl) => write!(f, "num {}", fl),
                ValueInner::Range(start, end) => write!(f, "range {}-{}", start, end),
                ValueInner::String(s) => write!(f, "str \"{}\"", s),
                ValueInner::List(lst) => {
                    write!(f, "list[")?;
                    for (i, val) in lst.borrow().iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", val)?;
                    }
                    write!(f, "]")
                }
                ValueInner::Map(map) => {
                    write!(f, "map{{")?;
                    for (i, (key, val)) in map.borrow().inner.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}: {}", key, val)?;
                    }
                    write!(f, "}}")
                }
            },
        }
    }
}

pub fn type_display(values: &[Value]) -> String {
    values
        .iter()
        .map(|item| format!("{}", item))
        .collect::<Vec<String>>()
        .join(", ")
}
