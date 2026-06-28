use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::mem::{self, ManuallyDrop};
use std::ptr;

use std::hash::Hash;

use dumpster::unsync::Gc;
use dumpster::{Trace, TraceWith, Visitor};
use foldhash::HashMap;

/// A tagged pointer that holds either a Gc<T> or a small integer.
///
/// We use the Least Significant Bit (LSB) as the tag:
/// 0 = Pointer
/// 1 = Integer
pub struct Value {
    /// Holds either a tagged SMI (LSB=1), null (0), or a transmuted Gc<ValueInner>.
    /// Cell is required so that dumpster's rehydration visitor can null out the Gc
    /// pointer through the shared `&self` reference in `TraceWith::accept`.
    ///
    /// Stored as `*mut ()` instead of `usize` to preserve pointer provenance,
    /// which lets Miri verify the correctness of our tagged-pointer scheme.
    bits: Cell<*mut ()>,
    _marker: PhantomData<Gc<ValueInner>>,
}

pub enum ValueRef<'a> {
    Null,
    Smi(i32),
    Float(f64),
    Range(&'a Range),
    String(&'a str),
    List(&'a RefCell<Vec<Value>>),
    Queue(&'a RefCell<VecDeque<Value>>),
    Map(&'a RefCell<Map>),
    Function(u32),
    Builtin(u32),
}

#[derive(Trace)]
pub enum ValueInner {
    Float(f64),
    Range(Range),
    String(String),
    List(RefCell<Vec<Value>>),
    Queue(RefCell<VecDeque<Value>>),
    Map(RefCell<Map>),
    Function(u32),
    Builtin(u32),
}

#[cfg(test)]
thread_local! {
    pub static VALUE_INNER_DROP_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
impl Drop for ValueInner {
    fn drop(&mut self) {
        VALUE_INNER_DROP_COUNT.with(|c| c.set(c.get() + 1));
    }
}

#[derive(Debug, PartialEq)]
pub struct Range {
    pub start: i64,
    pub end: i64,
}

pub struct Map {
    pub inner: HashMap<Value, Value>,
    pub iter_keys: Vec<Value>,
}

pub type OpResult = Option<Value>;

impl Value {
    const TAG_MASK: usize = 0b1;
    const INT_FLAG: usize = 0b1;

    pub const fn smi(val: i32) -> Self {
        let val_usize = val as usize;

        let bits = (val_usize << 32) | Self::INT_FLAG;

        Self {
            bits: Cell::new(ptr::without_provenance_mut(bits)),
            _marker: PhantomData,
        }
    }

    pub const fn null() -> Self {
        Self {
            bits: Cell::new(ptr::null_mut()),
            _marker: PhantomData,
        }
    }

    pub fn rc(gc: Gc<ValueInner>) -> Self {
        let ptr = unsafe { mem::transmute::<Gc<ValueInner>, *mut ()>(gc) };

        debug_assert!(
            ptr.addr() & Self::TAG_MASK == 0,
            "GC pointer was not properly aligned"
        );

        Self {
            bits: Cell::new(ptr),
            _marker: PhantomData,
        }
    }

    pub fn number(val: f64) -> Self {
        if val.fract() == 0.0 && val >= i32::MIN as f64 && val <= i32::MAX as f64 {
            Self::smi(val as i32)
        } else {
            Self::float(val)
        }
    }

    pub fn float(val: f64) -> Self {
        Self::rc(Gc::new(ValueInner::Float(val)))
    }

    pub fn int(val: i64) -> Self {
        if val < i32::MIN as i64 || val > i32::MAX as i64 {
            Self::float(val as f64)
        } else {
            Self::smi(val as i32)
        }
    }

    pub fn range(start: i64, end: i64) -> Self {
        Self::rc(Gc::new(ValueInner::Range(Range { start, end })))
    }

    pub fn string(val: String) -> Self {
        Self::rc(Gc::new(ValueInner::String(val)))
    }

    pub fn list(val: Vec<Value>) -> Self {
        Self::rc(Gc::new(ValueInner::List(RefCell::new(val))))
    }

    pub fn queue(val: VecDeque<Value>) -> Self {
        Self::rc(Gc::new(ValueInner::Queue(RefCell::new(val))))
    }

    pub fn bool(val: bool) -> Self {
        Self::smi(if val { 1 } else { 0 })
    }

    pub fn map(val: HashMap<Value, Value>) -> Self {
        Self::rc(Gc::new(ValueInner::Map(RefCell::new(Map {
            inner: val,
            iter_keys: Vec::new(),
        }))))
    }

    pub fn function(start_ip: u32) -> Self {
        Self::rc(Gc::new(ValueInner::Function(start_ip)))
    }

    pub fn builtin_ref(index: u32) -> Self {
        Self::rc(Gc::new(ValueInner::Builtin(index)))
    }

    pub fn is_smi(&self) -> bool {
        (self.bits.get().addr() & Self::TAG_MASK) == Self::INT_FLAG
    }

    pub fn is_gc(&self) -> bool {
        !self.is_null() && !self.is_smi()
    }

    pub fn is_string(&self) -> bool {
        matches!(self.as_value_ref(), ValueRef::String(_))
    }

    pub fn is_null(&self) -> bool {
        self.bits.get().is_null()
    }

    pub fn as_int(&self) -> Option<i64> {
        match self.as_value_ref() {
            ValueRef::Smi(i) => Some(i as i64),
            ValueRef::Float(f) => (f == f.floor()).then_some(f as i64),
            _ => None,
        }
    }

    pub fn as_number(&self) -> Option<f64> {
        match self.as_value_ref() {
            ValueRef::Smi(i) => Some(i as f64),
            ValueRef::Float(f) => Some(f),
            _ => None,
        }
    }

    pub fn as_value_ref(&self) -> ValueRef<'_> {
        if self.is_smi() {
            ValueRef::Smi((self.bits.get().addr() >> 32) as i32)
        } else if self.is_null() {
            ValueRef::Null
        } else {
            let gc = ManuallyDrop::new(unsafe {
                mem::transmute::<*mut (), Gc<ValueInner>>(self.bits.get())
            });
            // Safety: data lives as long as self (we hold a GC reference via bits)
            let ptr: *const ValueInner = &**gc;
            unsafe {
                match &*ptr {
                    ValueInner::Float(f) => ValueRef::Float(*f),
                    ValueInner::Range(r) => ValueRef::Range(r),
                    ValueInner::String(s) => ValueRef::String(s),
                    ValueInner::List(lst) => ValueRef::List(lst),
                    ValueInner::Queue(queue) => ValueRef::Queue(queue),
                    ValueInner::Map(map) => ValueRef::Map(map),
                    ValueInner::Function(ip) => ValueRef::Function(*ip),
                    ValueInner::Builtin(idx) => ValueRef::Builtin(*idx),
                }
            }
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
            _ => return None,
        };
        Some(res)
    }

    fn eq_impl(&self, other: &Self) -> Option<bool> {
        let res = match (self.as_value_ref(), other.as_value_ref()) {
            (ValueRef::Null, ValueRef::Null) => true,
            (ValueRef::Null, _) | (_, ValueRef::Null) => false,
            (ValueRef::Smi(a), ValueRef::Smi(b)) => a == b,
            (ValueRef::Smi(a), ValueRef::Float(b)) => a as f64 == b,
            (ValueRef::Float(a), ValueRef::Smi(b)) => a == b as f64,
            (ValueRef::Float(a), ValueRef::Float(b)) => a == b,
            (ValueRef::Range(a), ValueRef::Range(b)) => a == b,
            (ValueRef::String(a), ValueRef::String(b)) => a == b,
            (ValueRef::Function(a), ValueRef::Function(b)) => a == b,
            (ValueRef::Builtin(a), ValueRef::Builtin(b)) => a == b,
            _ => return None,
        };
        Some(res)
    }

    pub fn eq_(&self, other: &Self) -> OpResult {
        self.eq_impl(other)
            .map(|r| Value::smi(if r { 1 } else { 0 }))
    }

    pub fn neq(&self, other: &Self) -> OpResult {
        self.eq_impl(other)
            .map(|r| Value::smi(if !r { 1 } else { 0 }))
    }

    pub fn div(&self, other: &Self) -> OpResult {
        let res = match (self.as_value_ref(), other.as_value_ref()) {
            (ValueRef::Smi(a), ValueRef::Smi(b)) => {
                let res = a as f64 / b as f64;
                if res.fract() == 0.0 {
                    Value::int(res as i64)
                } else {
                    Value::float(res)
                }
            }
            (ValueRef::Smi(a), ValueRef::Float(b)) => Value::float(a as f64 / b),
            (ValueRef::Float(a), ValueRef::Smi(b)) => Value::float(a / b as f64),
            (ValueRef::Float(a), ValueRef::Float(b)) => Value::float(a / b),
            _ => return None,
        };
        Some(res)
    }

    pub fn is_falsy(&self) -> bool {
        match self.as_value_ref() {
            ValueRef::Null => true,
            ValueRef::Smi(i) => i == 0,
            ValueRef::Float(f) => f == 0.0,
            ValueRef::Range(_) => false,
            ValueRef::String(s) => s.is_empty(),
            ValueRef::List(lst) => lst.borrow().is_empty(),
            ValueRef::Queue(queue) => queue.borrow().is_empty(),
            ValueRef::Map(map) => map.borrow().inner.is_empty(),
            ValueRef::Function(_) | ValueRef::Builtin(_) => false,
        }
    }

    pub fn or(&self, other: &Self) -> OpResult {
        if self.is_falsy() {
            Some(other.clone())
        } else {
            Some(self.clone())
        }
    }

    pub fn and(&self, other: &Self) -> OpResult {
        if self.is_falsy() {
            Some(self.clone())
        } else {
            Some(other.clone())
        }
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
                    _ => return None,
                };
                Some(res)
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
                    _ => return None,
                };
                Some(Value::smi(if res { 1 } else { 0 }))
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

unsafe impl<V: Visitor> TraceWith<V> for Value {
    fn accept(&self, visitor: &mut V) -> Result<(), ()> {
        if self.is_gc() {
            // Reconstruct a Gc on the stack from our bits.
            // The visitor may null it out during dumpster's rehydration phase.
            let gc: Gc<ValueInner> =
                unsafe { mem::transmute::<*mut (), Gc<ValueInner>>(self.bits.get()) };
            visitor.visit_unsync(&gc);
            // Write back the (possibly nulled) bits so Drop won't use-after-free.
            let new_ptr: *mut () = unsafe { mem::transmute_copy(&gc) };
            self.bits.set(new_ptr);
            mem::forget(gc);
        }
        Ok(())
    }
}

unsafe impl<V: Visitor> TraceWith<V> for Map {
    fn accept(&self, visitor: &mut V) -> Result<(), ()> {
        for (k, v) in &self.inner {
            k.accept(visitor)?;
            v.accept(visitor)?;
        }
        for k in &self.iter_keys {
            k.accept(visitor)?;
        }
        Ok(())
    }
}

unsafe impl<V: Visitor> TraceWith<V> for Range {
    fn accept(&self, _visitor: &mut V) -> Result<(), ()> {
        Ok(())
    }
}

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
        Self::null()
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        if self.is_gc() {
            unsafe {
                let _gc: Gc<ValueInner> =
                    mem::transmute::<*mut (), Gc<ValueInner>>(self.bits.get());
            }
        }
    }
}

impl Clone for Value {
    fn clone(&self) -> Self {
        if self.is_gc() {
            let gc = ManuallyDrop::new(unsafe {
                mem::transmute::<*mut (), Gc<ValueInner>>(self.bits.get())
            });
            let cloned: Gc<ValueInner> = (*gc).clone();
            let ptr = unsafe { mem::transmute::<Gc<ValueInner>, *mut ()>(cloned) };
            Self {
                bits: Cell::new(ptr),
                _marker: PhantomData,
            }
        } else {
            Self {
                bits: Cell::new(self.bits.get()),
                _marker: PhantomData,
            }
        }
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_value_ref() {
            ValueRef::Null => write!(f, "NULL"),
            ValueRef::Smi(i) => write!(f, "{}", i),
            ValueRef::Float(fl) => write!(f, "{:.2}", fl),
            ValueRef::Range(r) => write!(f, "R{}-{}", r.start, r.end),
            ValueRef::String(s) => write!(f, "\"{}\"", s),
            ValueRef::List(lst) => {
                write!(f, "[")?;
                for (i, val) in lst.borrow().iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", val)?;

                    if i >= 5 {
                        write!(f, ", ...")?;
                        break;
                    }
                }
                write!(f, "]")
            }
            ValueRef::Queue(que) => {
                write!(f, "Queue[")?;
                for (i, val) in que.borrow().iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", val)?;

                    if i >= 5 {
                        write!(f, ", ...")?;
                        break;
                    }
                }
                write!(f, "]")
            }
            ValueRef::Map(map) => {
                write!(f, "{{")?;
                for (i, (key, val)) in map.borrow().inner.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}: {:?}", key, val)?;

                    if i >= 5 {
                        write!(f, ", ...")?;
                        break;
                    }
                }
                write!(f, "}}")
            }
            ValueRef::Function(ip) => write!(f, "<fn @{}>", ip),
            ValueRef::Builtin(idx) => write!(f, "<builtin #{}>", idx),
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(i) = self.as_int() {
            return write!(f, "int {}", i);
        }
        match self.as_value_ref() {
            ValueRef::Null => write!(f, "null"),
            ValueRef::Float(fl) => write!(f, "float {}", fl),
            ValueRef::Range(r) => write!(f, "range {}-{}", r.start, r.end),
            ValueRef::String(s) => write!(f, "str \"{}\"", s),
            ValueRef::Queue(que) => {
                write!(f, "queue[")?;
                for (i, val) in que.borrow().iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", val)?;

                    if i >= 5 {
                        write!(f, ", ...")?;
                        break;
                    }
                }
                write!(f, "]")
            }
            ValueRef::List(lst) => {
                write!(f, "list[")?;
                for (i, val) in lst.borrow().iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", val)?;

                    if i >= 5 {
                        write!(f, ", ...")?;
                        break;
                    }
                }
                write!(f, "]")
            }
            ValueRef::Map(map) => {
                write!(f, "map{{")?;
                for (i, (key, val)) in map.borrow().inner.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", key, val)?;

                    if i >= 5 {
                        write!(f, ", ...")?;
                        break;
                    }
                }
                write!(f, "}}")
            }
            ValueRef::Function(ip) => write!(f, "fn @{}", ip),
            ValueRef::Builtin(idx) => write!(f, "builtin #{}", idx),
            _ => unreachable!(),
        }
    }
}

pub fn type_display(values: &[Value]) -> String {
    let tstr = |value: &Value| match value.as_value_ref() {
        ValueRef::Null => "null",
        ValueRef::Smi(_) => "int",
        ValueRef::Float(_) => "float",
        ValueRef::Range(_) => "range",
        ValueRef::String(_) => "str",
        ValueRef::List(_) => "list",
        ValueRef::Queue(_) => "queue",
        ValueRef::Map(_) => "map",
        ValueRef::Function(_) => "function",
        ValueRef::Builtin(_) => "builtin",
    };
    values.iter().map(tstr).collect::<Vec<&str>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reset_drop_count() {
        VALUE_INNER_DROP_COUNT.with(|c| c.set(0));
    }
    fn drop_count() -> usize {
        VALUE_INNER_DROP_COUNT.with(|c| c.get())
    }

    /// Forces a `collect()` on drop. Declared first in a test so it drops last
    /// (after the test's `Value` locals), reclaiming their deferred GC boxes so
    /// Miri's leak checker stays clean. Cycle tests `collect()` explicitly
    /// instead, because they assert the drop count after collection.
    struct CollectGuard;
    impl Drop for CollectGuard {
        fn drop(&mut self) {
            dumpster::unsync::collect();
        }
    }

    // Cycle collection through tagged-pointer Values.
    // These tests assert that ValueInner::drop actually runs, proving
    // dumpster freed the cycle through our transmute-based representation.

    #[test]
    fn cycle_self_ref_list_is_freed() {
        reset_drop_count();
        {
            let list = Value::list(vec![]);
            match list.as_value_ref() {
                ValueRef::List(l) => l.borrow_mut().push(list.clone()),
                _ => unreachable!(),
            }
            // list -> list  (self-cycle via tagged pointer)
        }
        dumpster::unsync::collect();
        assert_eq!(
            drop_count(),
            1,
            "self-referencing list ValueInner must be freed"
        );
    }

    #[test]
    fn cycle_two_lists_are_freed() {
        reset_drop_count();
        {
            let a = Value::list(vec![]);
            let b = Value::list(vec![a.clone()]);
            match a.as_value_ref() {
                ValueRef::List(l) => l.borrow_mut().push(b.clone()),
                _ => unreachable!(),
            }
            // a -> b -> a  (mutual cycle)
        }
        dumpster::unsync::collect();
        assert_eq!(
            drop_count(),
            2,
            "both list ValueInners in mutual cycle must be freed"
        );
    }

    #[test]
    fn cycle_list_map_is_freed() {
        reset_drop_count();
        {
            let list = Value::list(vec![]);
            #[allow(clippy::mutable_key_type)]
            let mut m = HashMap::default();
            m.insert(Value::string("ref".into()), list.clone());
            let map = Value::map(m);
            match list.as_value_ref() {
                ValueRef::List(l) => l.borrow_mut().push(map.clone()),
                _ => unreachable!(),
            }
            // list -> map -> list  (cycle across Value types)
        }
        dumpster::unsync::collect();
        // list (1) + map (1) + the "ref" string key (1) = 3
        assert_eq!(
            drop_count(),
            3,
            "list, map, and string key ValueInners in cycle must be freed"
        );
    }

    #[test]
    fn non_cyclic_value_drop_counted() {
        let _collect = CollectGuard;
        // Sanity: normal (non-cyclic) GC values also hit the drop counter.
        reset_drop_count();
        {
            let _s = Value::string("hello".into());
            let _f = Value::float(1.5);
            let _l = Value::list(vec![Value::smi(1)]);
        }
        assert_eq!(
            drop_count(),
            3,
            "non-cyclic ValueInners must be dropped normally"
        );
    }

    // Size / layout guarantees

    #[test]
    fn value_is_pointer_sized() {
        assert_eq!(mem::size_of::<Value>(), mem::size_of::<usize>());
    }

    #[test]
    fn gc_is_pointer_sized() {
        assert_eq!(mem::size_of::<Gc<ValueInner>>(), mem::size_of::<usize>());
    }

    // Tagged pointer basics

    #[test]
    fn smi_tagging() {
        let v = Value::smi(42);
        assert!(v.is_smi() && !v.is_gc() && !v.is_null());
        assert_eq!(v.as_int(), Some(42));
    }

    #[test]
    fn null_tagging() {
        let v = Value::null();
        assert!(v.is_null() && !v.is_smi() && !v.is_gc());
    }

    #[test]
    fn gc_tagging() {
        let _collect = CollectGuard;
        let v = Value::float(3.1);
        assert!(v.is_gc() && !v.is_smi() && !v.is_null());
        assert_eq!(v.as_number(), Some(3.1));
    }

    #[test]
    fn clone_gc_string() {
        let _collect = CollectGuard;
        let a = Value::string("world".into());
        let b = a.clone();
        match (a.as_value_ref(), b.as_value_ref()) {
            (ValueRef::String(sa), ValueRef::String(sb)) => assert_eq!(sa, sb),
            _ => panic!("expected strings"),
        }
    }

    #[test]
    fn equality() {
        let _collect = CollectGuard;
        assert_eq!(Value::smi(1), Value::smi(1));
        assert_ne!(Value::smi(1), Value::smi(2));
        assert_eq!(Value::null(), Value::null());
        assert_eq!(Value::string("a".into()), Value::string("a".into()));
    }
}
