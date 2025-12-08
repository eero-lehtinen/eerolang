use std::ops::DerefMut;

use log::trace;

use crate::{
    ast_parser::fatal_generic,
    builtins::{ProgramFn, builtin_get},
    compiler::{
        ARG_REG_START, Addr, Compilation, FN_RETURN_VALUE_REG, Inst, RESERVED_REGS, RESULT_REG1,
        STACK_SIZE, SUCCESS_FLAG_REG, add_op, div_op, eq_op, gt_op, gte_op, is_zero, lt_op, lte_op,
        mul_op, neq_op, sub_op,
    },
    tokenizer::{MapValue, Operator, Token, Value, find_source_char_col, report_source_pos},
};

#[allow(dead_code)]
pub struct Vm<'a> {
    instructions: Vec<Inst>,
    ip_to_token: Vec<usize>,
    tokens: &'a [Token],
    inst_ptr: usize,
    stack_ptr: usize,
    sp_start: usize,
    memory: Vec<Value>,
    builtins: Vec<(ProgramFn, String)>,
}

fn placeholder_func(_: &[Value]) -> Result<Value, String> {
    Err("Placeholder function called".to_string())
}

#[allow(dead_code)]
impl<'a> Vm<'a> {
    pub fn new(ctx: Compilation<'a>) -> Self {
        let mut memory = vec![
            Value::Integer(0);
            RESERVED_REGS as usize + ctx.literals.len() + STACK_SIZE as usize
        ];
        for (lit_value, lit_reg) in ctx.literals.iter() {
            memory[lit_reg.get()] = lit_value.clone();
        }

        let mut builtins = vec![(placeholder_func as ProgramFn, String::new()); ctx.builtins.len()];
        for (name, (func, index, _)) in ctx.builtins.iter() {
            builtins[*index] = (*func, name.clone());
        }

        let sp = RESERVED_REGS as usize + ctx.literals.len();

        Vm {
            instructions: ctx.instructions,
            ip_to_token: ctx.ip_to_token,
            tokens: ctx.tokens,
            inst_ptr: 0,
            memory,
            builtins,
            stack_ptr: sp,
            sp_start: sp,
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

    #[inline]
    fn mem(&self, addr: Addr) -> usize {
        if addr.is_stack() {
            let offset = addr.get();
            // trace!(
            //     "Stack offset {}, pos: {}",
            //     offset,
            //     self.stack_ptr - offset as usize
            // );
            debug_assert!(self.stack_ptr - offset >= self.sp_start);
            self.stack_ptr - offset
        } else {
            addr.get()
        }
    }

    #[inline]
    fn mem_get(&self, addr: Addr) -> &Value {
        let pos = self.mem(addr);
        debug_assert!(pos < self.memory.len());
        // SAFETY: My memory offsets are surely correct.
        //         A little heap corruption never hurts :)
        unsafe { self.memory.get_unchecked(pos) }
    }

    #[inline]
    fn mem_set(&mut self, addr: Addr, value: Value) {
        let pos = self.mem(addr);
        debug_assert!(pos < self.memory.len());
        // SAFETY: Look up ^
        unsafe {
            *self.memory.get_unchecked_mut(pos) = value;
        }
    }

    pub fn run(&mut self, step_through: bool) {
        macro_rules! bop {
            ($fn:ident, $data:expr, $op:expr) => {{
                let l = self.mem_get($data.src1);
                let r = self.mem_get($data.src2);
                let res = $fn(|s| self.fatal(s), l, r);
                trace!(
                    "Binary op {} (at {}) {} {} (at {}) = {} (at {})",
                    l.dbg_display(),
                    $data.src1,
                    $op.dbg_display(),
                    r.dbg_display(),
                    $data.src2,
                    res.dbg_display(),
                    $data.dst
                );
                self.mem_set($data.dst, res);
            }};
        }
        while self.inst_ptr < self.instructions.len() {
            // trace!(
            //     "IP {}: {:?}",
            //     self.inst_ptr, self.instructions[self.inst_ptr]
            // );
            //
            let inst_ptr = self.inst_ptr;

            match &self.instructions[inst_ptr] {
                &Inst::LoadAddr { dst, src } => {
                    trace!(
                        "Load value {} from {} to {}",
                        self.mem_get(src).dbg_display(),
                        src,
                        dst
                    );
                    self.mem_set(dst, self.mem_get(src).clone());
                }
                &Inst::LoadInt { dst, value } => {
                    trace!("Load int {} to {}", value, dst);
                    self.mem_set(dst, value.into());
                }
                &Inst::LoadIterationKey { dst, src, index } => {
                    trace!(
                        "Load iteration key at index {} (at {}) of {} (at {}) to {}",
                        self.mem_get(index).dbg_display(),
                        index,
                        self.mem_get(src).dbg_display(),
                        src,
                        dst
                    );
                    let iterable = self.mem_get(src);
                    let index = self.mem_get(index);
                    let index = match index {
                        Value::Integer(i) => *i,
                        v => self.fatal(&format!(
                            "Expected (int) as index, got {:?}",
                            v.dbg_display()
                        )),
                    };

                    let key: Option<Value> = match iterable {
                        Value::List(l) => {
                            let list = l.borrow();
                            if index < 0 || index >= list.len() as i64 {
                                None
                            } else {
                                Some((index as i64).into())
                            }
                        }
                        Value::Range(r) => {
                            if index < 0 || index >= (r.end - r.start) {
                                None
                            } else {
                                Some((r.start + index).into())
                            }
                        }
                        Value::Map(map) => {
                            let map_borrow = map.borrow();
                            let list = map_borrow.iteration_keys.borrow();
                            if index < 0 || index >= list.len() as i64 {
                                None
                            } else {
                                Some(list[index as usize].clone())
                            }
                        }
                        v => self.fatal(&format!(
                            "Expected (list/range/map) as iterable, got {:?}",
                            v.dbg_display()
                        )),
                    };
                    self.mem_set(SUCCESS_FLAG_REG, (key.is_some() as i64).into());
                    if let Some(key) = key {
                        self.mem_set(dst, key);
                    }
                    trace!(
                        "  -> key: {:?}, success: {}",
                        self.mem_get(dst).dbg_display(),
                        self.mem_get(SUCCESS_FLAG_REG).dbg_display()
                    );
                }
                &Inst::LoadCollectionItem { dst, src, key } => {
                    trace!(
                        "Load collection item with key {} (at {}) from {} (at {}) to {}",
                        self.mem_get(key).dbg_display(),
                        key,
                        self.mem_get(src).dbg_display(),
                        src,
                        dst
                    );

                    let iterable = self.mem_get(src);
                    let key = self.mem_get(key);
                    if let Value::Range(_) = iterable {
                        self.mem_set(dst, key.clone());
                    } else {
                        self.mem_set(
                            dst,
                            builtin_get(&[iterable.clone(), key.clone()])
                                .expect("builtin_get failed"),
                        );
                    }
                }
                &Inst::InitMapIterationList { dst: src } => {
                    trace!(
                        "Init map iteration list for {} (at {})",
                        self.mem_get(src).dbg_display(),
                        src
                    );
                    // Other types get ignored
                    let map = self.mem_get(src);
                    if let Value::Map(map) = map {
                        let mut map_borrow = map.borrow_mut();
                        let MapValue {
                            inner,
                            iteration_keys,
                        } = map_borrow.deref_mut();

                        let mut iteration_keys_borrow = iteration_keys.borrow_mut();

                        if iteration_keys_borrow.is_empty() {
                            for key in inner.keys() {
                                iteration_keys_borrow.push(Value::from(key));
                            }
                        }
                    };
                }
                &Inst::AddStackPointer { value } => {
                    trace!("Add {} to stack pointer", value);
                    self.stack_ptr += value as usize;
                    if self.stack_ptr >= self.memory.len() {
                        self.fatal("Stack overflow");
                    }
                    debug_assert!(self.stack_ptr >= self.sp_start);
                }
                &Inst::SubStackPointer { value } => {
                    trace!("Subtract {} from stack pointer", value);
                    self.stack_ptr -= value as usize;
                    debug_assert!(self.stack_ptr >= self.sp_start);
                }
                Inst::Add(data) => bop!(add_op, data, Operator::Add),
                Inst::Sub(data) => bop!(sub_op, data, Operator::Sub),
                Inst::Mul(data) => bop!(mul_op, data, Operator::Mul),
                Inst::Div(data) => bop!(div_op, data, Operator::Div),
                Inst::Lt(data) => bop!(lt_op, data, Operator::Lt),
                Inst::Lte(data) => bop!(lte_op, data, Operator::Lte),
                Inst::Gt(data) => bop!(gt_op, data, Operator::Gt),
                Inst::Gte(data) => bop!(gte_op, data, Operator::Gte),
                Inst::Eq(data) => bop!(eq_op, data, Operator::Eq),
                Inst::Neq(data) => bop!(neq_op, data, Operator::Neq),
                &Inst::CallBuiltin {
                    dst,
                    func,
                    arg_count,
                } => {
                    trace!(
                        "Call builtin function {} with {} args, store result in {}",
                        self.builtins[func as usize].1, arg_count, dst
                    );
                    self.call_builtin(dst, func, arg_count);
                    trace!("  -> result: {:?}", self.mem_get(dst).dbg_display());
                }
                &Inst::Incr { dst } => {
                    trace!(
                        "Increment value {} (at {})",
                        self.mem_get(dst).dbg_display(),
                        dst
                    );
                    match self.mem_get(dst) {
                        Value::Integer(i) => {
                            self.mem_set(dst, Value::Integer(i + 1));
                        }
                        v => {
                            self.fatal(&format!("Expected (int), got {:?}", v.dbg_display()));
                        }
                    }
                }
                &Inst::Jump { target } => {
                    trace!("Jump from {} to {}", self.inst_ptr, target);
                    self.inst_ptr = target as usize;
                }
                &Inst::JumpAddr { target } => {
                    let target_value = self.mem_get(target);
                    let target_ip = match target_value {
                        Value::Integer(i) => *i as usize,
                        v => self.fatal(&format!(
                            "Expected (int) as jump address, got {:?}",
                            v.dbg_display()
                        )),
                    };
                    trace!(
                        "Jump from {} to {} (at {})",
                        self.inst_ptr, target_ip, target
                    );
                    self.inst_ptr = target_ip;
                }
                &Inst::JumpIfZero { target, cond } => {
                    trace!(
                        "JumpIfZero from {} to {} if {} (at {}) is zero",
                        self.inst_ptr,
                        target,
                        self.mem_get(cond).dbg_display(),
                        cond
                    );
                    let cond_value = &self.mem_get(cond);
                    let is_zero = is_zero(|s| self.fatal(s), cond_value);
                    if is_zero {
                        self.inst_ptr = target as usize;
                    }
                }
            }

            if step_through {
                let token = &self.tokens[self.ip_to_token[inst_ptr]];
                let char_col = find_source_char_col(token.line, token.byte_col);
                report_source_pos(token.line, char_col, 0);

                trace!(
                    "Regs: {}",
                    (RESULT_REG1.get() as u32..=FN_RETURN_VALUE_REG.get() as u32)
                        .chain(ARG_REG_START..ARG_REG_START + 6)
                        .map(|addr| {
                            format!(
                                "{}={}",
                                Addr::abs(addr),
                                self.mem_get(Addr::abs(addr)).dbg_display()
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                trace!(
                    "Stack: {}",
                    self.memory[self.sp_start..self.stack_ptr + 1]
                        .iter()
                        .map(|v| v.dbg_display())
                        .collect::<Vec<_>>()
                        .join(", ")
                );

                trace!("SP {}", self.stack_ptr - self.sp_start);

                let mut input = String::new();
                std::io::stdin().read_line(&mut input).unwrap();
            }

            if self.inst_ptr == inst_ptr {
                self.inst_ptr += 1;
            }
        }
    }

    #[inline]
    fn call_builtin(&mut self, dst: Addr, func: u32, arg_count: u8) {
        debug_assert!((func as usize) < self.builtins.len());
        // SAFETY: non-existent functions should be hard to call
        let func_impl = unsafe { self.builtins.get_unchecked(func as usize).0 };
        let args =
            &mut self.memory[ARG_REG_START as usize..ARG_REG_START as usize + arg_count as usize];
        let result = match func_impl(args) {
            Ok(v) => v,
            Err(e) => self.fatal(&format!("Error in function call: {}", e)),
        };
        self.mem_set(dst, result);
    }
}
