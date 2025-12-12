use std::ops::DerefMut;

use log::{info, trace};

use crate::{
    TOKENS,
    ast_parser::fatal_generic,
    builtins::{ProgramFn, builtin_get},
    compiler::{
        ARG_REG_START, ARG_REGS, Addr, Compilation, FN_RETURN_VALUE_REG, Inst, MEMORY_SIZE, OpCode,
        REGS_TO_STORE_ON_FN_CALL, RESERVED_REGS, RESULT_REG1, SUCCESS_FLAG_REG, binary_op_err,
    },
    tokenizer::{Operator, Token, find_source_char_col, report_source_pos},
    value::{Map, Value, ValueRef},
};

#[allow(dead_code)]
pub struct Vm<'a> {
    instructions: Vec<Inst>,
    ip_to_token: Vec<usize>,
    tokens: &'a [Token],
    inst_ptr: usize,
    stack_ptr: usize,
    sp_start: usize,
    sp_end: usize,
    memory: Vec<Value>,
    builtins: Vec<(ProgramFn, String)>,
}

fn placeholder_func(_: &[Value]) -> Result<Value, String> {
    Err("Placeholder function called".to_string())
}

#[allow(dead_code)]
impl<'a> Vm<'a> {
    pub fn new(ctx: Compilation<'a>) -> Self {
        let mut memory = vec![Value::default(); MEMORY_SIZE as usize];
        for (lit_value, lit_reg) in ctx.literals.iter() {
            memory[lit_reg.get()] = lit_value.clone();
        }

        let mut builtins = vec![(placeholder_func as ProgramFn, String::new()); ctx.builtins.len()];
        for (name, (func, index, _)) in ctx.builtins.iter() {
            builtins[*index] = (*func, name.clone());
        }

        let sp = RESERVED_REGS as usize;

        Vm {
            instructions: ctx.instructions,
            ip_to_token: ctx.ip_to_token,
            tokens: ctx.tokens,
            inst_ptr: 0,
            memory,
            builtins,
            stack_ptr: sp,
            sp_start: sp,
            sp_end: MEMORY_SIZE as usize - ctx.literals.len() - 1,
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

            debug_assert!(
                self.stack_ptr - offset >= RESERVED_REGS as usize
                    && self.stack_ptr - offset < self.sp_end
            );
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

    fn mem_swap(&mut self, addr1: Addr, addr2: Addr) {
        let pos1 = self.mem(addr1);
        let pos2 = self.mem(addr2);
        debug_assert!(pos1 < self.memory.len());
        debug_assert!(pos2 < self.memory.len());
        // SAFETY: Look up ^
        unsafe {
            let ptr1 = self.memory.get_unchecked_mut(pos1) as *mut Value;
            let ptr2 = self.memory.get_unchecked_mut(pos2) as *mut Value;
            std::ptr::swap(ptr1, ptr2);
        }
    }

    pub fn run(&mut self, step_through: bool) {
        macro_rules! bop {
            ($fn:ident, $data:expr, $op:expr) => {{
                let dst = Addr::from_raw($data.dst);
                let src1 = Addr::from_raw($data.src1);
                let src2 = Addr::from_raw($data.src2);
                let l = self.mem_get(src1);
                let r = self.mem_get(src2);
                if $op == Operator::Div {
                    if let Some(r_int) = r.as_int() {
                        if r_int == 0 {
                            self.fatal("Division by zero");
                        }
                    }
                }
                let res = match l.$fn(r) {
                    Ok(v) => v,
                    Err(e) => {
                        self.fatal(&binary_op_err(e, l, $op, r));
                    }
                };
                trace!(
                    "Binary op {} (at {}) {} {} (at {}) = {} (at {})",
                    l.dbg_display(),
                    src1,
                    $op,
                    r.dbg_display(),
                    src2,
                    res.dbg_display(),
                    dst
                );
                self.mem_set(dst, res);
            }};
        }

        while self.inst_ptr < self.instructions.len() {
            trace!("IP {}: {}", self.inst_ptr, self.instructions[self.inst_ptr]);

            let inst_ptr = self.inst_ptr;
            let inst = &self.instructions[inst_ptr];
            let Inst { opcode, args } = inst;

            match *opcode {
                OpCode::Nop => {
                    trace!("Nop");
                }
                OpCode::LoadAddr => {
                    let dst = Addr::from_raw(args.dst);
                    let src = Addr::from_raw(args.src1);
                    self.load_addr(dst, src);
                }
                OpCode::LoadInt => {
                    let dst = Addr::from_raw(args.dst);
                    let value = i32::from_ne_bytes(args.src1.to_ne_bytes());
                    trace!("Load int {} to {}", value, dst);
                    self.mem_set(dst, Value::smi(value));
                }
                OpCode::InitMapIter => {
                    let dst = Addr::from_raw(args.dst);
                    self.init_map_iter(dst);
                }
                OpCode::LoadIterKey => {
                    let dst = Addr::from_raw(args.dst);
                    let src = Addr::from_raw(args.src1);
                    let index = Addr::from_raw(args.src2);
                    self.load_iter_key(dst, src, index);
                }
                OpCode::LoadItem => {
                    let dst = Addr::from_raw(args.dst);
                    let src = Addr::from_raw(args.src1);
                    let key = Addr::from_raw(args.src2);
                    self.load_item(dst, src, key);
                }
                OpCode::AddStack => {
                    let value = args.dst;
                    self.add_stack(value);
                }
                OpCode::SubStack => {
                    let value = args.dst;
                    self.sub_stack(value);
                }
                OpCode::Add => bop!(add, args, Operator::Add),
                OpCode::Sub => bop!(sub, args, Operator::Sub),
                OpCode::Mul => bop!(mul, args, Operator::Mul),
                OpCode::Div => bop!(div, args, Operator::Div),
                OpCode::Lt => bop!(lt, args, Operator::Lt),
                OpCode::Lte => bop!(lte, args, Operator::Lte),
                OpCode::Gt => bop!(gt, args, Operator::Gt),
                OpCode::Gte => bop!(gte, args, Operator::Gte),
                OpCode::Eq => bop!(eq, args, Operator::Eq),
                OpCode::Neq => bop!(neq, args, Operator::Neq),
                OpCode::And => bop!(and, args, Operator::And),
                OpCode::Or => bop!(or, args, Operator::Or),
                OpCode::CallBuiltin => {
                    let func = args.dst;
                    let arg_count = args.src1 as u8;
                    self.call_builtin(func, arg_count);
                }
                OpCode::Incr => {
                    let dst = Addr::from_raw(args.dst);
                    self.incr(dst);
                }
                OpCode::SaveRegs => {
                    let arg_count = args.dst;
                    trace!("Save {} args and temporary registers to stack", arg_count);
                    for reg_addr in REGS_TO_STORE_ON_FN_CALL {
                        self.stack_ptr += 1;
                        self.mem_swap(*reg_addr, Addr::stack(0));
                    }
                    for arg_addr in ARG_REGS.iter().take(arg_count as usize) {
                        self.stack_ptr += 1;
                        self.mem_swap(*arg_addr, Addr::stack(0));
                    }
                }
                OpCode::RestoreRegs => {
                    let arg_count = args.dst;
                    trace!(
                        "Restore {} args and temporary registers from stack",
                        arg_count
                    );
                    for arg_addr in ARG_REGS.iter().take(arg_count as usize).rev() {
                        self.mem_swap(*arg_addr, Addr::stack(0));
                        self.stack_ptr -= 1;
                    }
                    for reg_addr in REGS_TO_STORE_ON_FN_CALL.iter().rev() {
                        self.mem_swap(*reg_addr, Addr::stack(0));
                        self.stack_ptr -= 1;
                    }
                }
                OpCode::Jump => {
                    let target = args.dst;
                    trace!("Jump from {} to {}", self.inst_ptr, target);
                    self.inst_ptr = target as usize;
                }
                OpCode::JumpAddr => {
                    let target = Addr::from_raw(args.dst);
                    let target_value = self.mem_get(target);
                    let Some(target_ip) = target_value.as_int() else {
                        self.fatal(&format!(
                            "Expected (int) as jump address, got {:?}",
                            target_value.dbg_display()
                        ));
                    };
                    trace!(
                        "Jump from {} to {} (at {})",
                        self.inst_ptr, target_ip, target
                    );
                    self.inst_ptr = target_ip as usize;
                }
                OpCode::JumpIfFalsy => {
                    let target = args.dst;
                    let cond = Addr::from_raw(args.src1);
                    trace!(
                        "JumpIfZero from {} to {} if {} (at {}) is zero",
                        self.inst_ptr,
                        target,
                        self.mem_get(cond).dbg_display(),
                        cond
                    );
                    let cond_value = &self.mem_get(cond);
                    if cond_value.is_falsy() {
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

    fn incr(&mut self, dst: Addr) {
        trace!(
            "Increment value {} (at {})",
            self.mem_get(dst).dbg_display(),
            dst
        );
        let v = self.mem_get(dst);
        if let Some(i) = v.as_int() {
            self.mem_set(dst, Value::int(i + 1));
        } else {
            self.fatal(&format!("Expected (int), got {:?}", v.dbg_display()));
        }
    }

    fn sub_stack(&mut self, value: u32) {
        trace!("Subtract {} from stack pointer", value);
        self.stack_ptr -= value as usize;
        debug_assert!(self.stack_ptr >= self.sp_start);
    }

    fn add_stack(&mut self, value: u32) {
        trace!("Add {} to stack pointer", value);
        self.stack_ptr += value as usize;
        if self.stack_ptr >= self.sp_end {
            self.fatal(&format!(
                "Stack overflow: stack pointer {} exceeds memory size {}",
                self.stack_ptr,
                self.memory.len()
            ));
        }
        debug_assert!(self.stack_ptr >= self.sp_start);
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
            "Regs: {}",
            (RESULT_REG1.get() as u32..=FN_RETURN_VALUE_REG.get() as u32)
                .chain(ARG_REG_START..ARG_REG_START + 6)
                .map(|addr| { format!("{}= {}", Addr::abs(addr), self.mem_get(Addr::abs(addr))) })
                .collect::<Vec<_>>()
                .join(", ")
        );
        info!(
            "Stack: {}",
            self.memory[self.sp_start..self.stack_ptr + 1]
                .iter()
                .map(|v| v.dbg_display())
                .collect::<Vec<_>>()
                .join(", ")
        );

        info!("SP {}", self.stack_ptr - self.sp_start);

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
    }

    fn load_addr(&mut self, dst: Addr, src: Addr) {
        trace!(
            "Load value {} from {} to {}",
            self.mem_get(src).dbg_display(),
            src,
            dst
        );
        self.mem_set(dst, self.mem_get(src).clone());
    }

    fn init_map_iter(&mut self, dst: Addr) {
        trace!(
            "Init map iteration list for {} (at {})",
            self.mem_get(dst).dbg_display(),
            dst
        );
        // Other types get ignored
        let value = self.mem_get(dst);
        if let ValueRef::Map(map) = value.as_value_ref() {
            let mut map = map.borrow_mut();
            let Map { inner, iter_keys } = map.deref_mut();
            if iter_keys.is_empty() {
                for key in inner.keys() {
                    iter_keys.push(key.clone());
                }
            }
        }
    }

    fn load_item(&mut self, dst: Addr, src: Addr, key: Addr) {
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
        match builtin_get(&[iterable.clone(), key.clone()]) {
            Ok(value) => {
                trace!("  -> loaded value: {:?}", value.dbg_display());
                self.mem_set(dst, value);
            }
            Err(e) => self.fatal(&format!("Wrong value in for loop iterable: {}", e)),
        }
    }

    fn load_iter_key(&mut self, dst: Addr, src: Addr, index: Addr) {
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
        let Some(index) = index.as_int() else {
            self.fatal(&format!(
                "Expected (int) as index, got {:?}",
                index.dbg_display()
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

        self.mem_set(
            SUCCESS_FLAG_REG,
            Value::smi(if key.is_some() { 1 } else { 0 }),
        );
        if let Some(key) = key {
            self.mem_set(dst, key);
        }
        trace!(
            "  -> key: {:?}, success: {}",
            self.mem_get(dst).dbg_display(),
            self.mem_get(SUCCESS_FLAG_REG).dbg_display()
        );
    }

    fn call_builtin(&mut self, func: u32, arg_count: u8) {
        trace!(
            "Call builtin function {} with {} args, store result in {}",
            self.builtins[func as usize].1, arg_count, FN_RETURN_VALUE_REG
        );
        debug_assert!((func as usize) < self.builtins.len());
        // SAFETY: non-existent functions should be hard to call
        let func_impl = unsafe { self.builtins.get_unchecked(func as usize).0 };
        let args =
            &mut self.memory[ARG_REG_START as usize..ARG_REG_START as usize + arg_count as usize];
        let result = match func_impl(args) {
            Ok(v) => v,
            Err(e) => self.fatal(&format!("Error in function call: {}", e)),
        };
        self.mem_set(FN_RETURN_VALUE_REG, result);
        trace!(
            "  -> result: {:?}",
            self.mem_get(FN_RETURN_VALUE_REG).dbg_display()
        );
    }
}
