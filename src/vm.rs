use std::ops::DerefMut;

use log::{info, trace};

use crate::{
    TOKENS,
    ast_parser::fatal_generic,
    builtins::{ProgramFn, builtin_get},
    compiler::{Compilation, binary_op_err},
    instructions::{ConstAddr, GlobalAddr, Inst, LocalAddr},
    tokenizer::{Operator, Token, find_source_char_col, report_source_pos},
    value::{Map, Value, ValueRef},
};

const STACK_SIZE: u32 = 2 << 12;

#[allow(dead_code)]
pub struct Vm<'a> {
    instructions: Vec<Inst>,
    ip_to_token: Vec<usize>,
    tokens: &'a [Token],
    inst_ptr: usize,
    eval_stack: Vec<Value>,
    eval_stack_ptr: usize,
    call_stack: Vec<Value>,
    call_stack_ptr: usize,
    globals: Vec<Value>,
    constants: Vec<Value>,
    builtins: Vec<(ProgramFn, String)>,
}

fn placeholder_func(_: &[Value]) -> Result<Value, String> {
    Err("Placeholder function called".to_string())
}

#[allow(dead_code)]
impl<'a> Vm<'a> {
    pub fn new(ctx: Compilation<'a>) -> Self {
        let mut builtins = vec![(placeholder_func as ProgramFn, String::new()); ctx.builtins.len()];
        for (name, (func, index, _)) in ctx.builtins.iter() {
            builtins[*index] = (*func, name.clone());
        }

        Vm {
            instructions: ctx.instructions,
            ip_to_token: ctx.ip_to_token,
            tokens: ctx.tokens,
            inst_ptr: 0,
            eval_stack: vec![const { Value::smi(0) }; STACK_SIZE as usize],
            eval_stack_ptr: 0,
            call_stack: vec![const { Value::smi(0) }; STACK_SIZE as usize],
            call_stack_ptr: 0,
            globals: vec![const { Value::smi(0) }; ctx.globals.len()],
            constants: ctx.constants.clone(),
            builtins,
        }
    }

    fn fatal(&self, msg: &str) -> ! {
        let token = &self.tokens[self.ip_to_token[self.inst_ptr]];
        fatal_generic(
            msg,
            &format!(
                "Fatal error during VM execution at inst {:?}",
                self.instructions[self.inst_ptr]
            ),
            token,
        )
    }

    fn constant(&self, addr: ConstAddr) -> &Value {
        debug_assert!((addr.0 as usize) < self.constants.len());
        // SAFETY: My things are correct.
        unsafe { self.constants.get_unchecked(addr.0 as usize) }
    }

    fn global(&self, addr: GlobalAddr) -> &Value {
        debug_assert!((addr.0 as usize) < self.globals.len());
        // SAFETY: My things are correct.
        unsafe { self.globals.get_unchecked(addr.0 as usize) }
    }

    fn local(&self, addr: LocalAddr) -> &Value {
        let pos = self.call_stack_ptr - addr.0 as usize;
        debug_assert!(pos < self.eval_stack.len());
        // SAFETY: My things are correct.
        unsafe { self.call_stack.get_unchecked(pos) }
    }

    fn pop_eval_to_local(&mut self, addr: LocalAddr) {
        trace!(
            "Pop value {} from stack to local {}",
            self.eval_stack(self.eval_stack_ptr).dbg_display(),
            addr
        );
        let local_pos = self.call_stack_ptr - addr.0 as usize;
        debug_assert!(local_pos < self.eval_stack.len());
        debug_assert!(self.eval_stack_ptr < self.eval_stack.len());
        // SAFETY: My things are correct.
        unsafe {
            std::ptr::swap(
                self.call_stack.get_unchecked_mut(local_pos) as *mut Value,
                self.eval_stack.get_unchecked_mut(self.eval_stack_ptr) as *mut Value,
            );
        }
        self.eval_stack_ptr -= 1;
    }

    fn pop_eval_to_global(&mut self, addr: GlobalAddr) {
        trace!(
            "Pop value {} from stack to global {}",
            self.eval_stack(self.eval_stack_ptr).dbg_display(),
            addr
        );
        let global_pos = addr.0 as usize;
        debug_assert!(global_pos < self.globals.len());
        debug_assert!(self.eval_stack_ptr < self.eval_stack.len());
        // SAFETY: My things are correct.
        unsafe {
            std::ptr::swap(
                self.globals.get_unchecked_mut(global_pos) as *mut Value,
                self.eval_stack.get_unchecked_mut(self.eval_stack_ptr) as *mut Value,
            );
        }
        self.eval_stack_ptr -= 1;
    }

    fn eval_stack(&self, offset: usize) -> &Value {
        let pos = self.eval_stack_ptr - offset;
        debug_assert!(pos < self.eval_stack.len());
        // SAFETY: My things are correct.
        unsafe { self.eval_stack.get_unchecked(pos) }
    }

    fn eval_stack_mut(&mut self, offset: usize) -> &mut Value {
        let pos = self.eval_stack_ptr - offset;
        debug_assert!(pos < self.eval_stack.len());
        // SAFETY: My things are correct.
        unsafe { self.eval_stack.get_unchecked_mut(pos) }
    }

    fn push_eval(&mut self, value: Value) {
        self.eval_stack_ptr += 1;
        if let Some(v) = self.eval_stack.get_mut(self.eval_stack_ptr) {
            *v = value;
        } else {
            self.fatal("Evaluation stack overflow");
        }
    }

    pub fn run(&mut self, step_through: bool) {
        macro_rules! bop {
            ($fn:ident, $data:expr, $op:expr) => {{
                let r = self.eval_stack(0);
                if $op == Operator::Div {
                    if let Some(f) = r.as_number() {
                        if f == 0.0 {
                            self.fatal("Division by zero");
                        }
                    }
                }
                let l = self.eval_stack(1);
                let res = match l.$fn(r) {
                    Some(v) => v,
                    None => {
                        self.fatal(&binary_op_err(l, $op, r));
                    }
                };
                trace!(
                    "Binary op {} {} {} = {} ",
                    l.dbg_display(),
                    $op,
                    r.dbg_display(),
                    res.dbg_display(),
                );
                self.eval_stack_ptr -= 1;
                self.eval_stack_mut(0).clone_from(&res);
            }};
        }

        while self.inst_ptr < self.instructions.len() {
            trace!("IP {}: {}", self.inst_ptr, self.instructions[self.inst_ptr]);

            let inst_ptr = self.inst_ptr;
            let inst = &self.instructions[inst_ptr];

            match *inst {
                Inst::Nop => {
                    trace!("Nop");
                }
                Inst::LoadConst(caddr) => {
                    let value = self.constant(caddr);
                    trace!(
                        "Push constant {} from {} to eval stack",
                        value.dbg_display(),
                        caddr,
                    );
                    self.push_eval(value.clone());
                }
                Inst::LoadGlobal(gaddr) => {
                    let value = self.global(gaddr);
                    trace!(
                        "Push global {} from {} to eval stack",
                        value.dbg_display(),
                        gaddr
                    );
                    self.push_eval(value.clone());
                }
                Inst::LoadLocal(laddr) => {
                    let value = self.local(laddr);
                    trace!(
                        "Push local {} from {} to eval stack",
                        value.dbg_display(),
                        laddr
                    );
                    self.push_eval(value.clone());
                }
                Inst::StoreGlobal(gaddr) => {
                    self.pop_eval_to_global(gaddr);
                }
                Inst::StoreLocal(laddr) => {
                    self.pop_eval_to_local(laddr);
                }
                Inst::Pop => {
                    trace!(
                        "Pop value {} from eval stack",
                        self.eval_stack(self.eval_stack_ptr).dbg_display()
                    );
                    self.eval_stack_ptr -= 1;
                }
                Inst::InitMapIter => {
                    let maybe_map = self.eval_stack(0);
                    match maybe_map.as_value_ref() {
                        ValueRef::Map(map_rc) => {
                            let mut map = map_rc.borrow_mut();
                            let Map { inner, iter_keys } = map.deref_mut();
                            if iter_keys.is_empty() {
                                for key in inner.keys() {
                                    iter_keys.push(key.clone());
                                }
                            }
                            trace!(
                                "Initialized map iterator with keys: {}",
                                iter_keys
                                    .iter()
                                    .map(|k| k.dbg_display())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            );
                        }
                        _ => {
                            trace!("InitMapIter called on non-map value, ignoring");
                        }
                    }
                    self.eval_stack_ptr -= 1;
                }
                Inst::LoadKey => {
                    let iterable = self.eval_stack(1);
                    let index_value = self.eval_stack(0);
                    let Some(index) = index_value.as_int() else {
                        self.fatal(&format!(
                            "Expected (int) as index, got {:?}",
                            index_value.dbg_display()
                        ));
                    };
                    let key = match iterable.as_value_ref() {
                        ValueRef::List(list_rc) => {
                            let list = list_rc.borrow();
                            if index < 0 || index >= list.len() as i64 {
                                None
                            } else {
                                Some(Value::int(index))
                            }
                        }
                        ValueRef::Range(start, end) => {
                            if index < 0 || index >= (end - start) {
                                None
                            } else {
                                Some(Value::int(start + index))
                            }
                        }
                        ValueRef::Map(map_rc) => {
                            let map = map_rc.borrow();
                            if index < 0 || index >= map.iter_keys.len() as i64 {
                                None
                            } else {
                                Some(map.iter_keys[index as usize].clone())
                            }
                        }
                        _ => self.fatal(&format!(
                            "Expected (list/range/map) as iterable, got {:?}",
                            iterable.dbg_display()
                        )),
                    };
                    self.eval_stack_ptr -= 1;

                    *self.eval_stack_mut(0) = match key {
                        Some(k) => {
                            trace!("LoadKey: found key: {:?}", k.dbg_display());
                            k
                        }
                        None => {
                            trace!("Index out of bounds in for loop iterable, using null");
                            Value::null()
                        }
                    };
                }
                Inst::LoadItem => {
                    match builtin_get(
                        &self.eval_stack[self.eval_stack_ptr - 1..=self.eval_stack_ptr],
                    ) {
                        Ok(value) => {
                            trace!("LoadKey: loaded value: {:?}", value.dbg_display());
                            self.eval_stack_ptr -= 1;
                            self.eval_stack_mut(0).clone_from(&value);
                        }
                        Err(e) => self.fatal(&format!("Wrong value in for loop iterable: {}", e)),
                    }
                }
                Inst::Incr => {
                    let v = self.eval_stack(0);
                    if let Some(i) = v.as_int() {
                        trace!(
                            "Increment value {} to {}",
                            v.dbg_display(),
                            Value::int(i + 1).dbg_display()
                        );
                        *self.eval_stack_mut(0) = Value::int(i + 1);
                    } else {
                        self.fatal(&format!("Expected (int), got {:?}", v.dbg_display()));
                    }
                }
                Inst::Add => bop!(add, args, Operator::Add),
                Inst::Sub => bop!(sub, args, Operator::Sub),
                Inst::Mul => bop!(mul, args, Operator::Mul),
                Inst::Div => bop!(div, args, Operator::Div),
                Inst::Lt => bop!(lt, args, Operator::Lt),
                Inst::Lte => bop!(lte, args, Operator::Lte),
                Inst::Gt => bop!(gt, args, Operator::Gt),
                Inst::Gte => bop!(gte, args, Operator::Gte),
                Inst::Eq => bop!(eq_, args, Operator::Eq),
                Inst::Neq => bop!(neq, args, Operator::Neq),
                Inst::CallBuiltin(index, args) => {
                    trace!(
                        "Call builtin function {} with {} args",
                        self.builtins[index as usize].1, args
                    );
                    debug_assert!((index as usize) < self.builtins.len());
                    // SAFETY: non-existent functions should be hard to call
                    let func_impl = unsafe { self.builtins.get_unchecked(index as usize).0 };
                    let arg_values = &mut self.eval_stack
                        [self.eval_stack_ptr - args as usize + 1..=self.eval_stack_ptr];
                    let result = match func_impl(arg_values) {
                        Ok(v) => v,
                        Err(e) => self.fatal(&format!("Error in function call: {}", e)),
                    };
                    self.eval_stack_ptr -= args as usize;
                    self.push_eval(result);
                    trace!("  -> result: {:?}", self.eval_stack(0).dbg_display());
                }
                Inst::Jump(target) => {
                    trace!("Jump from {} to {}", self.inst_ptr, target);
                    self.inst_ptr = target as usize;
                }
                Inst::JumpIfNull(target) => {
                    let cond_value = self.eval_stack(0);
                    trace!(
                        "JumpIfNull from {} to {} if {} is null",
                        self.inst_ptr,
                        target,
                        cond_value.dbg_display()
                    );
                    if cond_value.is_null() {
                        self.inst_ptr = target as usize;
                    }
                    self.eval_stack_ptr -= 1;
                }
                Inst::JumpIfFalsy(target) => {
                    let cond_value = self.eval_stack(0);
                    trace!(
                        "JumpIfFalsy from {} to {} if {} is falsy",
                        self.inst_ptr,
                        target,
                        cond_value.dbg_display()
                    );
                    if cond_value.is_falsy() {
                        self.inst_ptr = target as usize;
                    }
                }
                Inst::JumpIfTruthy(target) => {
                    let cond_value = self.eval_stack(0);
                    trace!(
                        "JumpIfTruthy from {} to {} if {} is truthy",
                        self.inst_ptr,
                        target,
                        cond_value.dbg_display()
                    );
                    if !cond_value.is_falsy() {
                        self.inst_ptr = target as usize;
                    }
                }
            }

            if step_through {
                self.step(inst_ptr);
            }

            if self.inst_ptr == inst_ptr {
                self.inst_ptr += 1;
            }
        }
    }

    fn step(&mut self, inst_ptr: usize) {
        let token = &self.tokens[self.ip_to_token[inst_ptr]];
        let char_col = find_source_char_col(token.line, token.byte_col);

        report_source_pos(
            TOKENS.get().unwrap(),
            token.line,
            char_col,
            token.byte_pos_start,
            token.byte_pos_end,
            1,
            colored::Color::BrightYellow,
        );

        info!(
            "Eval stack: {}",
            (0..=self.eval_stack_ptr)
                .map(|i| self.eval_stack(self.eval_stack_ptr - i).dbg_display())
                .collect::<Vec<_>>()
                .join(", ")
        );
        info!("Eval Stack Pointer: {}", self.eval_stack_ptr);

        info!(
            "Call stack: {}",
            (0..self.call_stack_ptr)
                .map(|i| self.call_stack[i].dbg_display())
                .collect::<Vec<_>>()
                .join(", ")
        );

        info!("Call Stack Pointer: {}", self.call_stack_ptr);

        info!(
            "Globals: {}",
            (0..self.globals.len())
                .map(|i| format!("{}: {}", i, self.globals[i].dbg_display()))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
    }
}
