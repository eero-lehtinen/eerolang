use std::{io::Write, ops::DerefMut};

use foldhash::HashMap;
use log::{info, trace};

use crate::{
    TOKENS,
    ast_parser::fatal_generic,
    builtins::{ProgramFn, builtin_get},
    compiler::{Compilation, binary_op_err},
    instructions::{ConstAddr, GlobalAddr, Inst, LocalAddr},
    tokenizer::{Operator, Token, TokenKind, find_source_char_col, report_source_pos},
    value::{Map, Value, ValueRef},
};

const STACK_SIZE: u32 = 2 << 12;

#[allow(dead_code)]
pub struct Vm<'a> {
    instructions: Vec<Inst>,
    ip_to_token: Vec<usize>,
    tokens: &'a [Token],
    inst_ptr: usize,
    stack: Vec<Value>,
    frame_ptr: usize,
    stack_ptr: usize,
    globals: Vec<Value>,
    global_names: Vec<&'a str>,
    constants: Vec<Value>,
    builtins: Vec<(ProgramFn, String)>,
    /// Maps ip to (function name, parameter names, local variable names)
    functions: HashMap<u32, (String, Vec<&'a str>, Vec<String>)>,
    current_fns: Vec<u32>,
    stop_at_line: Option<usize>,
    stepping: bool,
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

        let functions = ctx
            .functions
            .iter()
            .map(|(name, (ip, param_names, local_names))| {
                (
                    *ip,
                    (name.clone(), param_names.clone(), local_names.clone()),
                )
            })
            .collect();

        Vm {
            instructions: ctx.instructions,
            ip_to_token: ctx.ip_to_token,
            tokens: ctx.tokens,
            inst_ptr: 0,
            stack: vec![Value::default(); STACK_SIZE as usize],
            frame_ptr: 0,
            stack_ptr: 0,
            globals: vec![Value::default(); ctx.global_names.len()],
            global_names: ctx.global_names.clone(),
            constants: ctx.constants.clone(),
            builtins,
            functions,
            current_fns: Vec::new(),
            stop_at_line: None,
            stepping: false,
        }
    }

    fn fatal(&self, msg: &str) -> ! {
        let token_idx = if let Some(idx) = self.ip_to_token.get(self.inst_ptr) {
            *idx
        } else if self.inst_ptr > 0 {
            self.ip_to_token[self.inst_ptr - 1]
        } else {
            0
        };

        let instruction = self.instructions.get(self.inst_ptr);

        let token = &self.tokens[token_idx];
        fatal_generic(
            msg,
            &format!(
                "Fatal error during VM execution {}",
                if let Some(instruction) = instruction {
                    format!("({}) ", instruction)
                } else {
                    "".to_string()
                }
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

    fn global_mut(&mut self, addr: GlobalAddr) -> &mut Value {
        debug_assert!((addr.0 as usize) < self.globals.len());
        // SAFETY: My things are correct.
        unsafe { self.globals.get_unchecked_mut(addr.0 as usize) }
    }

    fn pos_to_local(&self, pos: usize) -> LocalAddr {
        LocalAddr((pos as isize - self.frame_ptr as isize) as i32)
    }

    fn local_pos(&self, addr: LocalAddr) -> usize {
        (self.frame_ptr as isize + addr.0 as isize) as usize
    }

    fn local(&self, addr: LocalAddr) -> &Value {
        let pos = self.local_pos(addr);
        debug_assert!(pos < self.stack.len());
        // SAFETY: My things are correct.
        unsafe { self.stack.get_unchecked(pos) }
    }

    fn local_mut(&mut self, addr: LocalAddr) -> &mut Value {
        let pos = self.local_pos(addr);
        debug_assert!(pos < self.stack.len());
        // SAFETY: My things are correct.
        unsafe { self.stack.get_unchecked_mut(pos) }
    }

    fn pop_to_local(&mut self, addr: LocalAddr) {
        trace!(
            "Pop value from stack to local {}={}",
            self.local_item_name(addr),
            self.stack(self.stack_ptr).dbg_display()
        );
        let local_pos = self.local_pos(addr);
        debug_assert!(local_pos < self.stack.len());
        debug_assert!(self.stack_ptr < self.stack.len());
        debug_assert!(local_pos != self.stack_ptr);
        // SAFETY: My things are correct.
        unsafe {
            std::ptr::swap(
                self.stack.get_unchecked_mut(local_pos) as *mut Value,
                self.stack.get_unchecked_mut(self.stack_ptr) as *mut Value,
            );
        }
        self.stack_ptr -= 1;
    }

    fn pop_to_global(&mut self, addr: GlobalAddr) {
        trace!(
            "Pop value from stack to global {}={}",
            self.global_names[addr.0 as usize],
            self.stack(self.stack_ptr).dbg_display(),
        );
        let global_pos = addr.0 as usize;
        debug_assert!(global_pos < self.globals.len());
        debug_assert!(self.stack_ptr < self.stack.len());
        // SAFETY: My things are correct.
        unsafe {
            std::ptr::swap(
                self.globals.get_unchecked_mut(global_pos) as *mut Value,
                self.stack.get_unchecked_mut(self.stack_ptr) as *mut Value,
            );
        }
        self.stack_ptr -= 1;
    }

    fn stack(&self, offset: usize) -> &Value {
        let pos = self.stack_ptr - offset;
        debug_assert!(pos < self.stack.len());
        // SAFETY: My things are correct.
        unsafe { self.stack.get_unchecked(pos) }
    }

    fn stack_mut(&mut self, offset: usize) -> &mut Value {
        let pos = self.stack_ptr - offset;
        debug_assert!(pos < self.stack.len());
        // SAFETY: My things are correct.
        unsafe { self.stack.get_unchecked_mut(pos) }
    }

    fn pop(&mut self) -> Value {
        let pos = self.stack_ptr;
        self.stack_ptr -= 1;
        debug_assert!(pos < self.stack.len());
        // SAFETY: My things are correct.
        unsafe { self.stack.get_unchecked(pos).clone() }
    }

    fn push(&mut self, value: Value) {
        self.stack_ptr += 1;
        if let Some(v) = self.stack.get_mut(self.stack_ptr) {
            *v = value;
        } else {
            self.fatal("Stack overflow");
        }
    }

    pub fn run(&mut self, step: bool) {
        self.stepping = step;
        macro_rules! bop {
            ($fn:ident, $data:expr, $op:expr) => {{
                let r = self.stack(0);
                if $op == Operator::Div {
                    if let Some(f) = r.as_number() {
                        if f == 0.0 {
                            self.fatal("Division by zero");
                        }
                    }
                }
                let l = self.stack(1);
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
                self.stack_ptr -= 1;
                *self.stack_mut(0) = res;
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
                        "Push constant {} from {} to stack",
                        value.dbg_display(),
                        caddr,
                    );
                    self.push(value.clone());
                }
                Inst::LoadGlobal(gaddr) => {
                    let value = self.global(gaddr);
                    trace!(
                        "Push global {}={} to stack",
                        self.global_names[gaddr.0 as usize],
                        value.dbg_display(),
                    );
                    self.push(value.clone());
                }
                Inst::LoadLocal(laddr) => {
                    let value = self.local(laddr);
                    trace!(
                        "Push local {}={} to stack",
                        self.local_item_name(laddr),
                        value.dbg_display()
                    );
                    self.push(value.clone());
                }
                Inst::StoreGlobal(gaddr) => {
                    self.pop_to_global(gaddr);
                }
                Inst::StoreLocal(laddr) => {
                    self.pop_to_local(laddr);
                }
                Inst::StoreGlobalKeep(gaddr) => {
                    let value = self.stack(0).clone();
                    trace!(
                        "Store global {}={}, keeping on stack",
                        self.global_names[gaddr.0 as usize],
                        value.dbg_display(),
                    );
                    *self.global_mut(gaddr) = value;
                }
                Inst::StoreLocalKeep(laddr) => {
                    let value = self.stack(0).clone();
                    trace!(
                        "Store local {}={}, keeping on stack",
                        self.local_item_name(laddr),
                        value.dbg_display()
                    );
                    *self.local_mut(laddr) = value;
                }
                Inst::Dup => {
                    let value = self.stack(0).clone();
                    trace!("Duplicate top stack value {}", value.dbg_display());
                    self.push(value);
                }
                Inst::Pop => {
                    trace!(
                        "Pop value {} from stack",
                        self.stack(self.stack_ptr).dbg_display()
                    );
                    self.stack_ptr -= 1;
                }
                Inst::InitMapIter => {
                    let maybe_map = self.stack(0);
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
                    self.stack_ptr -= 1;
                }
                Inst::LoadKey => {
                    let iterable = self.stack(1);
                    let index_value = self.stack(0);
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
                        ValueRef::Range(r) => {
                            if index >= (r.end - r.start).abs() {
                                None
                            } else {
                                Some(Value::int(r.start + index * (r.end - r.start).signum()))
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
                    self.stack_ptr -= 1;

                    *self.stack_mut(0) = match key {
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
                    match builtin_get(&self.stack[self.stack_ptr - 1..=self.stack_ptr]) {
                        Ok(value) => {
                            trace!("LoadKey: loaded value: {:?}", value.dbg_display());
                            self.stack_ptr -= 1;
                            self.stack_mut(0).clone_from(&value);
                        }
                        Err(e) => self.fatal(&format!("Wrong value in for loop iterable: {}", e)),
                    }
                }
                Inst::Incr => {
                    let v = self.stack(0);
                    if let Some(i) = v.as_int() {
                        trace!(
                            "Increment value {} to {}",
                            v.dbg_display(),
                            Value::int(i + 1).dbg_display()
                        );
                        *self.stack_mut(0) = Value::int(i + 1);
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
                Inst::And => bop!(and, args, Operator::And),
                Inst::Or => bop!(or, args, Operator::Or),
                Inst::CallBuiltin(index, args) => {
                    trace!(
                        "Call builtin function {} with {} args",
                        self.builtins[index as usize].1, args
                    );
                    debug_assert!((index as usize) < self.builtins.len());
                    // SAFETY: non-existent functions should be hard to call
                    let func_impl = unsafe { self.builtins.get_unchecked(index as usize).0 };
                    let arg_values =
                        &mut self.stack[self.stack_ptr - args as usize + 1..=self.stack_ptr];
                    let result = match func_impl(arg_values) {
                        Ok(v) => v,
                        Err(e) => self.fatal(&format!("Error in function call: {}", e)),
                    };
                    self.stack_ptr -= args as usize;
                    self.push(result);
                    trace!("  -> result: {:?}", self.stack(0).dbg_display());
                }
                Inst::Call(fn_ip, nlocals) => {
                    let return_ip = self.inst_ptr + 1;
                    self.push(Value::int(return_ip as i64));
                    self.push(Value::int(self.frame_ptr as i64));
                    self.frame_ptr = self.stack_ptr;
                    self.stack_ptr += nlocals as usize;
                    self.inst_ptr = fn_ip as usize;

                    if cfg!(debug_assertions) {
                        let (fn_name, param_names, local_names) =
                            self.functions.get(&fn_ip).unwrap();
                        self.current_fns.push(fn_ip);
                        trace!(
                            "Call function '{}' at {}, params: [{}], locals: [{}], return ip {}, new frame ptr {}",
                            fn_name,
                            fn_ip,
                            param_names.join(", "),
                            local_names.join(", "),
                            return_ip,
                            self.frame_ptr
                        );
                    }
                }
                Inst::Return(nargs) => {
                    let ret_value = self.pop();
                    self.stack_ptr = self.frame_ptr - nargs as usize - 2;
                    debug_assert!(self.stack_ptr < self.stack.len());
                    // SAFETY: My things are correct.
                    let old_frame_ptr = unsafe { self.stack.get_unchecked(self.frame_ptr) };
                    let frame_ptr = match old_frame_ptr.as_int() {
                        Some(fp) => fp as usize,
                        None => self.fatal(&format!(
                            "Corrupted frame pointer on return: {:?}",
                            old_frame_ptr.dbg_display()
                        )),
                    };
                    // SAFETY: My things are correct.
                    let return_ip_value = unsafe { self.stack.get_unchecked(self.frame_ptr - 1) };
                    let return_ip = match return_ip_value.as_int() {
                        Some(addr) => addr as usize,
                        None => self.fatal(&format!(
                            "Corrupted return address on return: {:?}",
                            return_ip_value.dbg_display()
                        )),
                    };
                    self.frame_ptr = frame_ptr;
                    self.push(ret_value);
                    self.inst_ptr = return_ip;

                    if cfg!(debug_assertions) {
                        let fn_ip = self.current_fns.pop().unwrap();
                        let (fn_name, _, _) = self.functions.get(&fn_ip).unwrap();
                        trace!(
                            "Return to IP {}, restored frame ptr {} from function '{}' at {}",
                            return_ip, frame_ptr, fn_name, fn_ip
                        );
                    }

                    debug_assert!(self.frame_ptr <= self.stack_ptr);
                }
                Inst::Jump(target) => {
                    trace!("Jump from {} to {}", self.inst_ptr, target);
                    self.inst_ptr = target as usize;
                }
                Inst::JumpIfNull(target) => {
                    let cond_value = self.stack(0);
                    trace!(
                        "JumpIfNull from {} to {} if {} is null",
                        self.inst_ptr,
                        target,
                        cond_value.dbg_display()
                    );
                    if cond_value.is_null() {
                        self.inst_ptr = target as usize;
                    }
                    self.stack_ptr -= 1;
                }
                Inst::JumpIfFalsy(target) => {
                    let cond_value = self.stack(0);
                    trace!(
                        "JumpIfFalsy from {} to {} if {} is falsy",
                        self.inst_ptr,
                        target,
                        cond_value.dbg_display()
                    );
                    if cond_value.is_falsy() {
                        self.inst_ptr = target as usize;
                    }
                    self.stack_ptr -= 1;
                }
                Inst::JumpIfTruthy(target) => {
                    let cond_value = self.stack(0);
                    trace!(
                        "JumpIfTruthy from {} to {} if {} is truthy",
                        self.inst_ptr,
                        target,
                        cond_value.dbg_display()
                    );
                    if !cond_value.is_falsy() {
                        self.inst_ptr = target as usize;
                    }
                    self.stack_ptr -= 1;
                }
            }

            if self.stepping {
                self.step(inst_ptr);
            }

            if self.inst_ptr == inst_ptr {
                self.inst_ptr += 1;
            }
        }

        if self.stack_ptr != 0 {
            self.fatal(&format!(
                "Stack not empty after execution ended, contents: {}",
                (1..=self.stack_ptr)
                    .map(|i| self.stack[i]
                        .dbg_display()
                        .chars()
                        .take(20)
                        .map(|c| if c == '\n' { '⏎' } else { c })
                        .collect::<String>())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }

    fn local_item_name(&self, addr: LocalAddr) -> String {
        if let Some((_, arg_names, local_names)) =
            self.current_fns.last().map(|ip| &self.functions[ip])
        {
            let addr = addr.0;
            if addr == 0 {
                "OLD_FP".to_string()
            } else if addr == -1 {
                "RET_ADDR".to_string()
            } else if addr < -1 {
                let arg_idx = arg_names.len() as i32 + addr + 1;
                if arg_idx < 0 {
                    return format!("{}", addr);
                }
                format!("ARG{}:{}", arg_idx, arg_names[arg_idx as usize])
            } else if addr >= 1 {
                let local_idx = addr as usize - 1;
                if local_idx >= local_names.len() {
                    return format!("{}", addr);
                }
                local_names[local_idx].clone()
            } else {
                format!("{}", addr)
            }
        } else {
            format!("{}", addr)
        }
    }

    fn step(&mut self, inst_ptr: usize) {
        if let Some(stop_line) = self.stop_at_line {
            let token = &self.tokens[self.ip_to_token[self.inst_ptr]];
            if token.line != stop_line {
                return;
            }
            self.stop_at_line = None;
        }

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

        // Get argument and local names for the current function frame

        let value_dbg_display = |v: &Value| {
            v.dbg_display()
                .chars()
                .take(20)
                .map(|c| if c == '\n' { '⏎' } else { c })
                .collect::<String>()
        };

        info!(
            "Stack: {}",
            (1..=self.stack_ptr)
                .map(|i| format!(
                    "{}={}",
                    self.local_item_name(self.pos_to_local(i)),
                    value_dbg_display(&self.stack[i])
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );

        info!(
            "Stack Ptr: {}, Frame Ptr: {}",
            self.stack_ptr, self.frame_ptr
        );

        info!(
            "Globals: {}",
            (0..self.globals.len())
                .map(|i| format!(
                    "{}={}",
                    self.global_names[i],
                    value_dbg_display(&self.globals[i])
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );

        print!("Press enter or type a line number: ");
        let mut input = String::new();
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut input).unwrap();
        println!();

        let input = input.trim();
        if let Ok(line_num) = input.parse::<usize>() {
            let res = self
                .tokens
                .iter()
                .find(|t| t.kind != TokenKind::Comment && t.line >= line_num - 1);

            let token = match res {
                Some(t) => t,
                None => self.tokens.last().unwrap(),
            };

            self.stop_at_line = Some(token.line);
        }
    }
}
