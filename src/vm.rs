use std::ops::DerefMut;

use log::{info, trace};

use crate::{
    TOKENS,
    ast_parser::fatal_generic,
    builtins::{ProgramFn, builtin_get},
    compiler::{Compilation, ConstAddr, GlobalAddr, Inst, LocalAddr, binary_op_err},
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

    // fn mem_swap(&mut self, addr1: Addr, addr2: Addr) {
    //     let pos1 = self.mem(addr1);
    //     let pos2 = self.mem(addr2);
    //     debug_assert!(pos1 < self.eval_stack.len());
    //     debug_assert!(pos2 < self.eval_stack.len());
    //     // SAFETY: Look up ^
    //     unsafe {
    //         let ptr1 = self.eval_stack.get_unchecked_mut(pos1) as *mut Value;
    //         let ptr2 = self.eval_stack.get_unchecked_mut(pos2) as *mut Value;
    //         std::ptr::swap(ptr1, ptr2);
    //     }
    // }

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
                } // Inst::

                  // OpCode::LoadAddr => {
                  //     let dst = Addr::from_raw(args.dst);
                  //     let src = Addr::from_raw(args.src1);
                  //     self.load_addr(dst, src);
                  // }
                  // OpCode::LoadInt => {
                  //     let dst = Addr::from_raw(args.dst);
                  //     let value = i32::from_ne_bytes(args.src1.to_ne_bytes());
                  //     trace!("Load int {} to {}", value, dst);
                  //     self.mem_set(dst, Value::smi(value));
                  // }
                  // OpCode::Push => {
                  //     let src = Addr::from_raw(args.dst);
                  //     trace!(
                  //         "Push value {} (at {}) to stack",
                  //         self.mem_get(src).dbg_display(),
                  //         src
                  //     );
                  //     self.stack_ptr += 1;
                  //     self.mem_set(Addr::stack(0), self.mem_get(src).clone());
                  // }
                  // OpCode::Pop => {
                  //     let dst = Addr::from_raw(args.dst);
                  //     trace!(
                  //         "Pop value {} from stack to {}",
                  //         self.mem_get(Addr::stack(0)).dbg_display(),
                  //         dst
                  //     );
                  //     self.mem_swap(dst, Addr::stack(0));
                  //     self.stack_ptr -= 1;
                  // }
                  // OpCode::InitMapIter => {
                  //     let dst = Addr::from_raw(args.dst);
                  //     self.init_map_iter(dst);
                  // }
                  // OpCode::LoadIterKey => {
                  //     let dst = Addr::from_raw(args.dst);
                  //     let src = Addr::from_raw(args.src1);
                  //     let index = Addr::from_raw(args.src2);
                  //     self.load_iter_key(dst, src, index);
                  // }
                  // OpCode::LoadItem => {
                  //     let dst = Addr::from_raw(args.dst);
                  //     let src = Addr::from_raw(args.src1);
                  //     let key = Addr::from_raw(args.src2);
                  //     self.load_item(dst, src, key);
                  // }
                  // OpCode::AddStack => {
                  //     let value = args.dst;
                  //     self.add_stack(value);
                  // }
                  // OpCode::SubStack => {
                  //     let value = args.dst;
                  //     self.sub_stack(value);
                  // }

                  // OpCode::And => bop!(and, args, Operator::And),
                  // OpCode::Or => bop!(or, args, Operator::Or),
                  // OpCode::CallBuiltin => {
                  //     let func = args.dst;
                  //     let arg_count = args.src1 as u8;
                  //     self.call_builtin(func, arg_count);
                  // }
                  // OpCode::Incr => {
                  //     let dst = Addr::from_raw(args.dst);
                  //     self.incr(dst);
                  // }
                  // OpCode::SaveRegs => {
                  //     let arg_count = args.dst;
                  //     trace!("Save {} args and temporary registers to stack", arg_count);
                  //     for reg_addr in REGS_TO_STORE_ON_FN_CALL {
                  //         self.stack_ptr += 1;
                  //         self.mem_swap(*reg_addr, Addr::stack(0));
                  //     }
                  //     for arg_addr in ARG_REGS.iter().take(arg_count as usize) {
                  //         self.stack_ptr += 1;
                  //         self.mem_swap(*arg_addr, Addr::stack(0));
                  //     }
                  // }
                  // OpCode::RestoreRegs => {
                  //     let arg_count = args.dst;
                  //     trace!(
                  //         "Restore {} args and temporary registers from stack",
                  //         arg_count
                  //     );
                  //     for arg_addr in ARG_REGS.iter().take(arg_count as usize).rev() {
                  //         self.mem_swap(*arg_addr, Addr::stack(0));
                  //         self.stack_ptr -= 1;
                  //     }
                  //     for reg_addr in REGS_TO_STORE_ON_FN_CALL.iter().rev() {
                  //         self.mem_swap(*reg_addr, Addr::stack(0));
                  //         self.stack_ptr -= 1;
                  //     }
                  // }
                  // OpCode::Jump => {
                  //     let target = args.dst;
                  //     trace!("Jump from {} to {}", self.inst_ptr, target);
                  //     self.inst_ptr = target as usize;
                  // }
                  // OpCode::JumpAddr => {
                  //     let target = Addr::from_raw(args.dst);
                  //     let target_value = self.mem_get(target);
                  //     let Some(target_ip) = target_value.as_int() else {
                  //         self.fatal(&format!(
                  //             "Expected (int) as jump address, got {:?}",
                  //             target_value.dbg_display()
                  //         ));
                  //     };
                  //     trace!(
                  //         "Jump from {} to {} (at {})",
                  //         self.inst_ptr, target_ip, target
                  //     );
                  //     self.inst_ptr = target_ip as usize;
                  // }
                  // OpCode::JumpIfFalsy => {
                  //     let target = args.dst;
                  //     let cond = Addr::from_raw(args.src1);
                  //     trace!(
                  //         "JumpIfZero from {} to {} if {} (at {}) is zero",
                  //         self.inst_ptr,
                  //         target,
                  //         self.mem_get(cond).dbg_display(),
                  //         cond
                  //     );
                  //     let cond_value = &self.mem_get(cond);
                  //     if cond_value.is_falsy() {
                  //         self.inst_ptr = target as usize;
                  //     }
                  // }
            }

            if step_through {
                self.step(inst_ptr);
            }

            if self.inst_ptr == inst_ptr {
                self.inst_ptr += 1;
            }
        }
    }
    //
    // fn incr(&mut self, dst: Addr) {
    //     trace!(
    //         "Increment value {} (at {})",
    //         self.mem_get(dst).dbg_display(),
    //         dst
    //     );
    //     let v = self.mem_get(dst);
    //     if let Some(i) = v.as_int() {
    //         self.mem_set(dst, Value::int(i + 1));
    //     } else {
    //         self.fatal(&format!("Expected (int), got {:?}", v.dbg_display()));
    //     }
    // }
    //
    // fn sub_stack(&mut self, value: u32) {
    //     trace!("Subtract {} from stack pointer", value);
    //     self.call_stack_ptr -= value as usize;
    //     debug_assert!(self.call_stack_ptr >= self.sp_start);
    // }
    //
    // fn add_stack(&mut self, value: u32) {
    //     trace!("Add {} to stack pointer", value);
    //     self.call_stack_ptr += value as usize;
    //     if self.call_stack_ptr >= self.sp_end {
    //         self.fatal(&format!(
    //             "Stack overflow: stack pointer {} exceeds memory size {}",
    //             self.call_stack_ptr,
    //             self.eval_stack.len()
    //         ));
    //     }
    //     debug_assert!(self.call_stack_ptr >= self.sp_start);
    // }
    //
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
    //
    // fn load_addr(&mut self, dst: Addr, src: Addr) {
    //     trace!(
    //         "Load value {} from {} to {}",
    //         self.mem_get(src).dbg_display(),
    //         src,
    //         dst
    //     );
    //     self.mem_set(dst, self.mem_get(src).clone());
    // }
    //
    // fn init_map_iter(&mut self, dst: Addr) {
    //     trace!(
    //         "Init map iteration list for {} (at {})",
    //         self.mem_get(dst).dbg_display(),
    //         dst
    //     );
    //     // Other types get ignored
    //     let value = self.mem_get(dst);
    //     if let ValueRef::Map(map) = value.as_value_ref() {
    //         let mut map = map.borrow_mut();
    //         let Map { inner, iter_keys } = map.deref_mut();
    //         if iter_keys.is_empty() {
    //             for key in inner.keys() {
    //                 iter_keys.push(key.clone());
    //             }
    //         }
    //     }
    // }
    //
    // fn load_item(&mut self, dst: Addr, src: Addr, key: Addr) {
    //     trace!(
    //         "Load collection item with key {} (at {}) from {} (at {}) to {}",
    //         self.mem_get(key).dbg_display(),
    //         key,
    //         self.mem_get(src).dbg_display(),
    //         src,
    //         dst
    //     );
    //     let iterable = self.mem_get(src);
    //     let key = self.mem_get(key);
    //     match builtin_get(&[iterable.clone(), key.clone()]) {
    //         Ok(value) => {
    //             trace!("  -> loaded value: {:?}", value.dbg_display());
    //             self.mem_set(dst, value);
    //         }
    //         Err(e) => self.fatal(&format!("Wrong value in for loop iterable: {}", e)),
    //     }
    // }
    //
    // fn load_iter_key(&mut self, dst: Addr, src: Addr, index: Addr) {
    //     trace!(
    //         "Load iteration key at index {} (at {}) of {} (at {}) to {}",
    //         self.mem_get(index).dbg_display(),
    //         index,
    //         self.mem_get(src).dbg_display(),
    //         src,
    //         dst
    //     );
    //     let iterable = self.mem_get(src);
    //     let index = self.mem_get(index);
    //     let Some(index) = index.as_int() else {
    //         self.fatal(&format!(
    //             "Expected (int) as index, got {:?}",
    //             index.dbg_display()
    //         ));
    //     };
    //
    //     let key = match iterable.as_value_ref() {
    //         ValueRef::List(list_rc) => {
    //             let list = list_rc.borrow();
    //             if index < 0 || index >= list.len() as i64 {
    //                 None
    //             } else {
    //                 Some(Value::int(index))
    //             }
    //         }
    //         ValueRef::Range(start, end) => {
    //             if index < 0 || index >= (end - start) {
    //                 None
    //             } else {
    //                 Some(Value::int(start + index))
    //             }
    //         }
    //         ValueRef::Map(map_rc) => {
    //             let map = map_rc.borrow();
    //             if index < 0 || index >= map.iter_keys.len() as i64 {
    //                 None
    //             } else {
    //                 Some(map.iter_keys[index as usize].clone())
    //             }
    //         }
    //         _ => self.fatal(&format!(
    //             "Expected (list/range/map) as iterable, got {:?}",
    //             iterable.dbg_display()
    //         )),
    //     };
    //
    //     self.mem_set(
    //         SUCCESS_FLAG_REG,
    //         Value::smi(if key.is_some() { 1 } else { 0 }),
    //     );
    //     if let Some(key) = key {
    //         self.mem_set(dst, key);
    //     }
    //     trace!(
    //         "  -> key: {:?}, success: {}",
    //         self.mem_get(dst).dbg_display(),
    //         self.mem_get(SUCCESS_FLAG_REG).dbg_display()
    //     );
    // }
    //
    // fn call_builtin(&mut self, func: u32, arg_count: u8) {
    //     trace!(
    //         "Call builtin function {} with {} args, store result in {}",
    //         self.builtins[func as usize].1, arg_count, FN_RETURN_VALUE_REG
    //     );
    //     debug_assert!((func as usize) < self.builtins.len());
    //     // SAFETY: non-existent functions should be hard to call
    //     let func_impl = unsafe { self.builtins.get_unchecked(func as usize).0 };
    //     let args = &mut self.eval_stack
    //         [ARG_REG_START as usize..ARG_REG_START as usize + arg_count as usize];
    //     let result = match func_impl(args) {
    //         Ok(v) => v,
    //         Err(e) => self.fatal(&format!("Error in function call: {}", e)),
    //     };
    //     self.mem_set(FN_RETURN_VALUE_REG, result);
    //     trace!(
    //         "  -> result: {:?}",
    //         self.mem_get(FN_RETURN_VALUE_REG).dbg_display()
    //     );
    // }
}
