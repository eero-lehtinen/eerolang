use std::fmt::Display;

use foldhash::{HashMap, HashMapExt};
use log::trace;

use crate::{
    ast_parser::{AstNode, AstNodeKind, fatal_generic},
    builtins::{ArgsRequred, ProgramFn, all_builtins},
    tokenizer::{Operator, Token},
    value::{OpError, OpResult, Value},
};

/// This is very likely slow, it should just be a different instruction to use the stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Addr(u32);

const FLAG_BIT: u32 = 1 << 31;
const DATA_MASK: u32 = !FLAG_BIT;

impl Addr {
    pub const fn abs(val: u32) -> Self {
        if val & FLAG_BIT != 0 {
            panic!("Absolute address too big, would overlap with flag bit");
        }
        Addr(val)
    }

    pub const fn stack(val: u32) -> Self {
        if val & FLAG_BIT != 0 {
            panic!("Stack address too big, would overlap with flag bit");
        }
        Addr(val | FLAG_BIT)
    }

    pub fn is_stack(&self) -> bool {
        (self.0 & FLAG_BIT) != 0
    }

    #[allow(dead_code)]
    pub fn is_abs(&self) -> bool {
        (self.0 & FLAG_BIT) == 0
    }

    pub fn get(&self) -> usize {
        (self.0 & DATA_MASK) as usize
    }

    fn raw(&self) -> u32 {
        self.0
    }

    pub fn from_raw(raw: u32) -> Self {
        Addr(raw)
    }
}

impl Display for Addr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_stack() {
            write!(f, "S{:<4}", self.get())
        } else if let Some(name) = [
            ("RES1", RESULT_REG1),
            ("RES2", RESULT_REG2),
            ("SUC", SUCCESS_FLAG_REG),
            ("RETA", FN_CALL_RETURN_ADDR_REG),
            ("RET", FN_RETURN_VALUE_REG),
        ]
        .iter()
        .find_map(|(name, addr)| if *addr == *self { Some(*name) } else { None })
        {
            write!(f, "{:<5}", name)
        } else if self.get() >= ARG_REG_START as usize
            && self.get() < ARG_REG_START as usize + ARG_REG_COUNT as usize
        {
            write!(f, "ARG{:<2}", self.get() - ARG_REG_START as usize)
        } else {
            write!(f, "A{:<4}", self.get())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCode {
    Nop,
    LoadAddr,
    LoadInt,
    InitMapIter,
    LoadIterKey,
    LoadItem,
    AddStack,
    SubStack,
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Gt,
    Lte,
    Gte,
    Eq,
    Neq,
    Incr,
    CallBuiltin,
    SaveRegs,
    RestoreRegs,
    Jump,
    JumpAddr,
    JumpIfZero,
}

impl Display for OpCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone)]
pub struct OpArgs {
    pub dst: u32,
    pub src1: u32,
    pub src2: u32,
}

#[derive(Debug, Clone)]
pub struct Inst {
    pub opcode: OpCode,
    pub args: OpArgs,
}

impl Inst {
    fn new(opcode: OpCode, dst: u32, src1: u32, src2: u32) -> Self {
        Inst {
            opcode,
            args: OpArgs { dst, src1, src2 },
        }
    }

    pub fn nop() -> Self {
        Self::new(OpCode::Nop, 0, 0, 0)
    }

    pub fn load_addr(dst: Addr, src: Addr) -> Self {
        Self::new(OpCode::LoadAddr, dst.raw(), src.raw(), 0)
    }

    pub fn load_int(dst: Addr, value: i32) -> Self {
        Self::new(
            OpCode::LoadInt,
            dst.raw(),
            u32::from_ne_bytes(value.to_ne_bytes()),
            0,
        )
    }

    pub fn init_map_iteration_list(dst: Addr) -> Self {
        Self::new(OpCode::InitMapIter, dst.raw(), 0, 0)
    }

    pub fn load_iteration_key(dst: Addr, src: Addr, index: Addr) -> Self {
        Self::new(OpCode::LoadIterKey, dst.raw(), src.raw(), index.raw())
    }
    pub fn load_collection_item(dst: Addr, src: Addr, key: Addr) -> Self {
        Self::new(OpCode::LoadItem, dst.raw(), src.raw(), key.raw())
    }

    pub fn add_stack_pointer(value: u32) -> Self {
        Self::new(OpCode::AddStack, value, 0, 0)
    }

    pub fn sub_stack_pointer(value: u32) -> Self {
        Self::new(OpCode::SubStack, value, 0, 0)
    }

    pub fn binary_op(op: Operator, dst: Addr, src1: Addr, src2: Addr) -> Self {
        let opcode = match op {
            Operator::Add => OpCode::Add,
            Operator::Sub => OpCode::Sub,
            Operator::Mul => OpCode::Mul,
            Operator::Div => OpCode::Div,
            Operator::Lt => OpCode::Lt,
            Operator::Gt => OpCode::Gt,
            Operator::Lte => OpCode::Lte,
            Operator::Gte => OpCode::Gte,
            Operator::Eq => OpCode::Eq,
            Operator::Neq => OpCode::Neq,
        };

        Self::new(opcode, dst.raw(), src1.raw(), src2.raw())
    }

    pub fn incr(dst: Addr) -> Self {
        Self::new(OpCode::Incr, dst.raw(), 0, 0)
    }

    pub fn call_builtin(dst: Addr, func: u32, arg_count: u8) -> Self {
        Self::new(OpCode::CallBuiltin, dst.raw(), func, arg_count as u32)
    }

    pub fn save_regs() -> Self {
        Self::new(OpCode::SaveRegs, 0, 0, 0)
    }

    pub fn restore_regs() -> Self {
        Self::new(OpCode::RestoreRegs, 0, 0, 0)
    }

    pub fn jump(target: u32) -> Self {
        Self::new(OpCode::Jump, target, 0, 0)
    }

    pub fn jump_addr(target: Addr) -> Self {
        Self::new(OpCode::JumpAddr, target.raw(), 0, 0)
    }

    pub fn jump_if_zero(target: u32, cond: Addr) -> Self {
        Self::new(OpCode::JumpIfZero, target, cond.raw(), 0)
    }

    fn set_incr_dst(&mut self, dst: Addr) {
        assert_eq!(self.opcode, OpCode::Incr);
        self.args.dst = dst.raw();
    }

    fn set_jump_target(&mut self, target_ip: u32) {
        assert!(
            matches!(self.opcode, OpCode::Jump | OpCode::JumpIfZero),
            "Can only set jump target on jump instructions"
        );

        self.args.dst = target_ip;
    }
}

impl Display for Inst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:<10} ", &format!("{}", self.opcode))?;
        match self.opcode {
            OpCode::Nop => {}
            OpCode::LoadAddr => write!(
                f,
                "{} {}",
                Addr::from_raw(self.args.dst),
                Addr::from_raw(self.args.src1)
            )?,
            OpCode::LoadInt => write!(
                f,
                "{} {:<5}",
                Addr::from_raw(self.args.dst),
                i32::from_ne_bytes(self.args.src1.to_ne_bytes())
            )?,
            OpCode::InitMapIter => write!(f, "{}", Addr::from_raw(self.args.dst))?,
            OpCode::LoadIterKey => write!(
                f,
                "{} {} {}",
                Addr::from_raw(self.args.dst),
                Addr::from_raw(self.args.src1),
                Addr::from_raw(self.args.src2)
            )?,
            OpCode::LoadItem => write!(
                f,
                "{} {} {}",
                Addr::from_raw(self.args.dst),
                Addr::from_raw(self.args.src1),
                Addr::from_raw(self.args.src2)
            )?,
            OpCode::AddStack | OpCode::SubStack => write!(f, "{:<5}", self.args.dst)?,
            OpCode::Add
            | OpCode::Sub
            | OpCode::Mul
            | OpCode::Div
            | OpCode::Lt
            | OpCode::Gt
            | OpCode::Lte
            | OpCode::Gte
            | OpCode::Eq
            | OpCode::Neq => write!(
                f,
                "{} {} {}",
                Addr::from_raw(self.args.dst),
                Addr::from_raw(self.args.src1),
                Addr::from_raw(self.args.src2)
            )?,
            OpCode::Incr => write!(f, "{}", Addr::from_raw(self.args.dst))?,
            OpCode::CallBuiltin => write!(
                f,
                "{} {:<5} {:<5}",
                Addr::from_raw(self.args.dst),
                self.args.src1,
                self.args.src2
            )?,
            OpCode::SaveRegs => {}
            OpCode::RestoreRegs => {}
            OpCode::Jump => write!(f, "{:<5}", self.args.dst)?,
            OpCode::JumpAddr => write!(f, "{}", Addr::from_raw(self.args.dst))?,
            OpCode::JumpIfZero => {
                write!(f, "{:<5} {}", self.args.dst, Addr::from_raw(self.args.src1))?
            }
        }

        Ok(())
    }
}

pub const ARG_REG_START: u32 = 0;
pub const ARG_REG_COUNT: u32 = 8;
const fn reg(n: u32) -> Addr {
    Addr::abs(ARG_REG_START + ARG_REG_COUNT + n)
}
pub const RESULT_REG1: Addr = reg(0);
pub const RESULT_REG2: Addr = reg(1);
pub const SUCCESS_FLAG_REG: Addr = reg(2);
pub const FN_CALL_RETURN_ADDR_REG: Addr = reg(3);
pub const FN_RETURN_VALUE_REG: Addr = reg(4);
pub const RESERVED_REGS: u32 = ARG_REG_START + ARG_REG_COUNT + 3;
pub const STACK_SIZE: u32 = 2 << 12;

pub struct Compilation<'a> {
    pub instructions: Vec<Inst>,
    pub literals: Vec<(Value, Addr)>,
    pub builtins: HashMap<String, (ProgramFn, usize, ArgsRequred)>,
    pub functions: HashMap<String, (u32, ArgsRequred)>,
    pub tokens: &'a [Token],
    pub ip_to_token: Vec<usize>,
    pub scope_var_decls: Vec<Vec<(&'a str, usize)>>,
    cur_stack_ptr_offset: u32,
}

impl<'a> Compilation<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        let mut builtins = HashMap::new();
        for (i, (name, func, args)) in all_builtins().iter().enumerate() {
            builtins.insert(name.to_string(), (*func, i, *args));
        }
        Compilation {
            instructions: Vec::new(),
            literals: Vec::new(),
            builtins,
            functions: HashMap::new(),
            tokens,
            ip_to_token: Vec::new(),
            scope_var_decls: Vec::new(),
            cur_stack_ptr_offset: 0,
        }
    }

    fn fatal(&self, msg: &str, node: &AstNode) -> ! {
        let token = &self.tokens[node.token_idx];
        fatal_generic(msg, "Fatal error during compilation", token)
    }

    fn cur_inst_ptr(&self) -> u32 {
        self.instructions.len() as u32
    }

    fn inst_mut(&mut self, ip: u32) -> &mut Inst {
        &mut self.instructions[ip as usize]
    }

    fn variable_offset(&mut self, name: &str, node: &AstNode) -> Addr {
        trace!("{:?}", self.scope_var_decls);

        let pos = self.tokens[node.token_idx].byte_pos_start;

        let mut frame_ptr = self.cur_stack_ptr_offset;
        for decls in self.scope_var_decls.iter().rev() {
            frame_ptr -= decls.len() as u32;
            for (i, (dname, dpos)) in decls.iter().enumerate() {
                if *dname == name {
                    if *dpos > pos {
                        self.fatal(
                            &format!("Variable '{}' used before initialization", name),
                            node,
                        );
                    }
                    let offset = self.cur_stack_ptr_offset - frame_ptr - i as u32 - 1;
                    trace!("variable offset: {} for variable '{}'", offset, name);
                    return Addr::stack(offset);
                }
            }
        }

        self.fatal(
            &format!("Variable '{}' used before declaration", name),
            node,
        );
    }

    fn make_literal(&mut self, value: &Value) -> Addr {
        let addr = Addr::abs(self.literals.len() as u32 + RESERVED_REGS);
        self.literals.push((value.clone(), addr));
        addr
    }

    fn push_instruction(&mut self, inst: Inst, node: &AstNode) {
        self.instructions.push(inst);
        self.ip_to_token.push(node.token_idx);
    }

    fn compile_assignment(&mut self, node: &AstNode, ctx: &mut Context) {
        let (var, expr) = match &node.kind {
            AstNodeKind::DeclareAssign(var, expr) => (var, expr),
            AstNodeKind::Assign(var, expr) => (var, expr),
            _ => unreachable!(),
        };
        let var_addr = self.variable_offset(var, node);
        let (expr_addr, _) = self.compile_expression(expr, var_addr, ctx);
        if expr_addr != var_addr {
            self.push_instruction(Inst::load_addr(var_addr, expr_addr), node);
        }
    }

    fn compile_expression(
        &mut self,
        expr: &AstNode,
        dst_suggestion: Addr,
        ctx: &mut Context,
    ) -> (Addr, Option<Value>) {
        match &expr.kind {
            AstNodeKind::Literal(value) => (self.make_literal(value), Some(value.clone())),
            AstNodeKind::Variable(name) => (self.variable_offset(name, expr), None),
            AstNodeKind::BinaryOp(left, op, right) => {
                let (laddr, lval) = self.compile_expression(left, RESULT_REG1, ctx);
                let (raddr, rval) = self.compile_expression(right, RESULT_REG2, ctx);

                // Constant folding for literals
                if let (Some(lit_left), Some(lit_right)) = (lval, rval) {
                    let folded_value = match binary_op(&lit_left, *op, &lit_right) {
                        Ok(v) => v,
                        Err(e) => {
                            self.fatal(&binary_op_err(e, &lit_left, *op, &lit_right), expr);
                        }
                    };
                    return (self.make_literal(&folded_value), Some(folded_value));
                }

                self.push_instruction(Inst::binary_op(*op, dst_suggestion, laddr, raddr), expr);
                (dst_suggestion, None)
            }
            AstNodeKind::FunctionCall(..) => {
                self.compile_function_call(dst_suggestion, expr, ctx);
                (dst_suggestion, None)
            }
            _ => todo!(),
        }
    }

    fn compile_function_definition(&mut self, node: &'a AstNode, ctx: &Context) {
        let AstNodeKind::FunctionDefinition(name, args, body) = &node.kind else {
            unreachable!();
        };

        if all_builtins().iter().any(|(n, _, _)| n == name) {
            self.fatal(
                &format!("Cannot redefine built-in function '{}'", name),
                node,
            );
        }
        if self.functions.contains_key(name) {
            self.fatal(&format!("Function '{}' is already defined", name), node);
        }

        let fn_skip_jump_ip = self.cur_inst_ptr();
        // Function instructions are defined "in the middle" of the instructions so we need to skip
        // over it to make top level code work correctly.
        self.push_instruction(Inst::jump(0), node);

        let fn_start_ip = self.cur_inst_ptr();

        let args_required = ArgsRequred::Exact(args.len() as u32);
        self.functions
            .insert(name.clone(), (fn_start_ip, args_required));

        let mut fn_ctx = self.block_start(body, ctx, Some(node), None);

        // Store return address in a stack variable.
        let return_addr_var_addr = self.variable_offset(Self::FN_CALL_RETURN_ADDR_VAR, node);
        self.push_instruction(
            Inst::load_addr(return_addr_var_addr, FN_CALL_RETURN_ADDR_REG),
            node,
        );

        // Load arguments from argument registers to stack variables
        for (arg_idx, arg) in args.iter().enumerate() {
            let arg_name = arg.get_var_name().expect("Parsed correctly");
            let arg_addr = self.variable_offset(arg_name, arg);
            let arg_reg = Addr::abs(ARG_REG_START + arg_idx as u32);
            self.push_instruction(Inst::load_addr(arg_addr, arg_reg), node);
        }

        self.compile_block(body, &mut fn_ctx);

        // Default return value is 1
        self.push_instruction(Inst::load_int(FN_RETURN_VALUE_REG, 1), node);

        // Store return address back to the return address register.
        // TODO: This might not be needed if we clean up the stack in the caller.
        self.push_instruction(
            Inst::load_addr(FN_CALL_RETURN_ADDR_REG, return_addr_var_addr),
            node,
        );

        // Clean up stack frame.
        self.block_end(body, &fn_ctx);

        // Jump back to return address.
        self.push_instruction(Inst::jump_addr(FN_CALL_RETURN_ADDR_REG), node);

        let fn_end_ip = self.cur_inst_ptr();
        self.inst_mut(fn_skip_jump_ip).set_jump_target(fn_end_ip);
    }

    fn compile_return(&mut self, node: &'a AstNode, ctx: &mut Context) {
        let AstNodeKind::Return(expr) = &node.kind else {
            unreachable!();
        };
        let (expr_addr, _) = self.compile_expression(expr, FN_RETURN_VALUE_REG, ctx);
        if expr_addr != FN_RETURN_VALUE_REG {
            self.push_instruction(Inst::load_addr(FN_RETURN_VALUE_REG, expr_addr), node);
        }

        // Store return address back to the return address register.
        // TODO: This might not be needed if we clean up the stack in the caller.
        let return_addr_var_addr = self.variable_offset(Self::FN_CALL_RETURN_ADDR_VAR, node);
        self.push_instruction(
            Inst::load_addr(FN_CALL_RETURN_ADDR_REG, return_addr_var_addr),
            node,
        );

        // Clean up stack frame.
        let sub_sp = self.cur_stack_ptr_offset - ctx.fn_frame_ptr;
        if sub_sp > 0 {
            self.push_instruction(Inst::sub_stack_pointer(sub_sp), node);
        }

        // Jump back to return address.
        self.push_instruction(Inst::jump_addr(FN_CALL_RETURN_ADDR_REG), node);
    }

    fn compile_set_function_args(&mut self, args: &[AstNode], ctx: &mut Context) {
        if args.len() > ARG_REG_COUNT as usize {
            self.fatal("Too many arguments in function call", &args[0]);
        }
        for (i, arg) in args.iter().enumerate() {
            let arg_reg = Addr::abs(ARG_REG_START + i as u32);
            let (res_addr, _) = self.compile_expression(arg, arg_reg, ctx);
            if res_addr != arg_reg {
                self.push_instruction(Inst::load_addr(arg_reg, res_addr), arg);
            }
        }
    }

    fn compile_function_call(&mut self, dst: Addr, node: &AstNode, ctx: &mut Context) {
        let AstNodeKind::FunctionCall(name, args) = &node.kind else {
            unreachable!();
        };

        macro_rules! unexpected_args {
            ($args_req:expr) => {
                self.fatal(
                    &format!(
                        "Function '{}' expects {} arguments, got {}",
                        name,
                        $args_req.describe(),
                        args.len(),
                    ),
                    node,
                );
            };
        }

        if let Some((_, func_index, args_req)) = self.builtins.get(name).cloned() {
            if !args_req.matches(args.len()) {
                unexpected_args!(args_req);
            }

            self.compile_set_function_args(args, ctx);

            self.push_instruction(
                Inst::call_builtin(dst, func_index as u32, args.len() as u8),
                node,
            );
        } else if let Some(&(fn_start_ip, args_req)) = self.functions.get(name) {
            if !args_req.matches(args.len()) {
                unexpected_args!(args_req);
            }

            self.compile_set_function_args(args, ctx);

            // Store return address (placeholder)
            let load_ret_addr_ip = self.cur_inst_ptr();
            self.push_instruction(Inst::nop(), node);

            // Store temporaries to survive the function call.
            self.push_instruction(Inst::save_regs(), node);
            // self.push_instruction(Inst::add_stack_pointer(2), node);
            // self.push_instruction(Inst::load_addr(Addr::stack(1), RESULT_REG1), node);
            // self.push_instruction(Inst::load_addr(Addr::stack(0), RESULT_REG2), node);

            // Jump to the function.
            self.push_instruction(Inst::jump(fn_start_ip), node);

            // Store return address (placeholder)
            *self.inst_mut(load_ret_addr_ip) =
                Inst::load_int(FN_CALL_RETURN_ADDR_REG, self.cur_inst_ptr() as i32);

            // Restore temporaries after the function call.
            self.push_instruction(Inst::load_addr(RESULT_REG2, Addr::stack(0)), node);
            self.push_instruction(Inst::load_addr(RESULT_REG1, Addr::stack(1)), node);
            self.push_instruction(Inst::sub_stack_pointer(2), node);

            // Load return value to the correct location.
            self.push_instruction(Inst::load_addr(dst, FN_RETURN_VALUE_REG), node);
        } else {
            self.fatal(&format!("Undefined function: {}", name), node);
        }
    }

    fn compile_for_loop(&mut self, node: &'a AstNode, ctx: &Context) {
        let AstNodeKind::ForLoop(key, item, collection, body) = &node.kind else {
            unreachable!();
        };
        let mut for_ctx = self.block_start(body, ctx, None, Some(node));
        for_ctx.loop_frame_ptr = ctx.block_frame_ptr;
        for_ctx.loop_stack_ptr = self.cur_stack_ptr_offset;

        let iterable_addr = self.variable_offset(Self::FOR_ITERABLE_TEMP_VAR, node);
        let (addr, _) = self.compile_expression(collection, iterable_addr, &mut for_ctx);
        if addr != iterable_addr {
            self.push_instruction(Inst::load_addr(iterable_addr, addr), collection);
        };

        self.push_instruction(Inst::init_map_iteration_list(iterable_addr), node);

        let index_addr = self.variable_offset(Self::FOR_INDEX_TEMP_VAR, node);
        self.push_instruction(Inst::load_int(index_addr, 0), node);

        let for_load_key_ip = self.cur_inst_ptr();

        let (key_var_name, key_node) = if let Some(key_node) = key {
            (
                key_node.get_var_name().expect("Parsed correctly"),
                key_node.as_ref(),
            )
        } else {
            (Self::FOR_KEY_TEMP_VAR, node)
        };
        let key_addr = self.variable_offset(key_var_name, key_node);
        self.push_instruction(
            Inst::load_iteration_key(key_addr, iterable_addr, index_addr),
            key_node,
        );

        let for_exit_jump_ip = self.cur_inst_ptr();
        // Placeholder
        self.push_instruction(Inst::jump_if_zero(0, SUCCESS_FLAG_REG), node);

        if let Some(item_node) = item {
            let item_var_name = item_node.get_var_name().expect("Parsed correctly");
            let item_addr = self.variable_offset(item_var_name, item_node);
            self.push_instruction(
                Inst::load_collection_item(item_addr, iterable_addr, key_addr),
                item_node,
            );
        }

        self.compile_block(body, &mut for_ctx);

        self.push_instruction(Inst::incr(index_addr), node);
        self.push_instruction(Inst::jump(for_load_key_ip), node);

        let loop_end_ip = self.cur_inst_ptr();
        self.inst_mut(for_exit_jump_ip).set_jump_target(loop_end_ip);

        for continue_ip in &for_ctx.loop_continues {
            self.inst_mut(*continue_ip).set_incr_dst(index_addr);
            self.inst_mut(*continue_ip + 1)
                .set_jump_target(for_load_key_ip);
        }

        self.block_end(body, &for_ctx);

        let loop_end_after_sp_reset_ip = self.cur_inst_ptr();

        for break_ip in &for_ctx.loop_breaks {
            self.inst_mut(*break_ip)
                .set_jump_target(loop_end_after_sp_reset_ip);
        }
    }

    fn compile_continue(&mut self, node: &'a AstNode, ctx: &mut Context) {
        let sub_sp = self.cur_stack_ptr_offset - ctx.loop_stack_ptr;
        if sub_sp > 0 {
            self.push_instruction(Inst::sub_stack_pointer(sub_sp), node);
        }
        ctx.loop_continues.push(self.cur_inst_ptr());
        // Placeholder
        self.push_instruction(Inst::incr(Addr::abs(0)), node);
        // Placeholder
        self.push_instruction(Inst::jump(0), node);
    }

    fn compile_break(&mut self, node: &'a AstNode, ctx: &mut Context) {
        let sub_sp = self.cur_stack_ptr_offset - ctx.loop_frame_ptr;
        if sub_sp > 0 {
            self.push_instruction(Inst::sub_stack_pointer(sub_sp), node);
        }
        ctx.loop_breaks.push(self.cur_inst_ptr());
        // Placeholder
        self.push_instruction(Inst::jump(0), node);
    }

    fn compile_if_statement(&mut self, node: &'a AstNode, ctx: &mut Context) {
        let AstNodeKind::IfStatement(condition, block, else_block) = &node.kind else {
            unreachable!();
        };
        let (cond_addr, cond_val) = self.compile_expression(condition, RESULT_REG1, ctx);

        let const_cond_true = cond_val.map(|v| {
            !is_zero(&v).unwrap_or_else(|| {
                self.fatal(
                    "Condition expression in if statement must be an integer",
                    condition,
                )
            })
        });

        if let Some(const_cond_true) = const_cond_true {
            if const_cond_true {
                self.compile_block_full(block, ctx);
            } else if let Some(else_block) = else_block {
                self.compile_block_full(else_block, ctx);
            }
            return;
        }

        let if_jump_ip = self.cur_inst_ptr();
        // Placeholder
        self.push_instruction(Inst::jump_if_zero(0, cond_addr), node);

        self.compile_block_full(block, ctx);

        if let Some(else_block) = else_block {
            let else_jump_ip = self.cur_inst_ptr();
            // Placeholder
            self.push_instruction(Inst::jump(0), node);

            let else_start_ip = self.cur_inst_ptr();
            self.instructions[if_jump_ip as usize].set_jump_target(else_start_ip);

            self.compile_block_full(else_block, ctx);

            let after_else_ip = self.cur_inst_ptr();
            self.instructions[else_jump_ip as usize].set_jump_target(after_else_ip);
        } else {
            let after_if_ip = self.cur_inst_ptr();
            self.instructions[if_jump_ip as usize].set_jump_target(after_if_ip);
        }
    }

    // Temporary variable to store function call return address. Needed for nested function calls.
    const FN_CALL_RETURN_ADDR_VAR: &'static str = "__fn_call_return_addr";

    // If the iterable is an expression, it needs to be stored somwhere.
    const FOR_ITERABLE_TEMP_VAR: &'static str = "__for_iterable_temp";
    // Index needs to be stored somewhere.
    const FOR_INDEX_TEMP_VAR: &'static str = "__for_index_temp";
    // Even if not assigned to a variable, the key needs to be stored somewhere.
    const FOR_KEY_TEMP_VAR: &'static str = "__for_key_temp";

    fn block_start(
        &mut self,
        node: &'a AstNode,
        prev_ctx: &Context,
        fn_node: Option<&'a AstNode>,
        loop_node: Option<&'a AstNode>,
    ) -> Context {
        let AstNodeKind::Block(nodes) = &node.kind else {
            self.fatal("Expected block node", node);
        };

        let mut cur_scope_var_decls: Vec<(&'a str, usize)> = Vec::new();
        macro_rules! add_decl_node {
            ($decl:expr) => {
                let var_name = $decl.get_var_name().expect("Parsed correctly");
                if cur_scope_var_decls.iter().any(|(v, _)| *v == var_name) {
                    self.fatal(
                        &format!("Variable '{}' already declared in this scope", var_name),
                        $decl,
                    );
                }
                let token = &self.tokens[$decl.token_idx];
                cur_scope_var_decls.push((var_name, token.byte_pos_start));
            };
        }

        if let Some(node) = fn_node {
            let AstNodeKind::FunctionDefinition(_, args, _) = &node.kind else {
                panic!("Should be parsed correctly");
            };

            let token = &self.tokens[node.token_idx];
            cur_scope_var_decls.push((Self::FN_CALL_RETURN_ADDR_VAR, token.byte_pos_start));

            for arg in args {
                add_decl_node!(arg);
            }
        }

        // Add loop variable declarations
        if let Some(loop_node) = loop_node {
            let AstNodeKind::ForLoop(key, item, _, _) = &loop_node.kind else {
                panic!("Should be parsed correctly");
            };

            let token = &self.tokens[loop_node.token_idx];
            // These are always not named by the user.
            cur_scope_var_decls.extend_from_slice(&[
                (Self::FOR_ITERABLE_TEMP_VAR, token.byte_pos_start),
                (Self::FOR_INDEX_TEMP_VAR, token.byte_pos_start),
            ]);

            // This is needed but allowed to be underscore by the user.
            if let Some(key_node) = key {
                add_decl_node!(key_node);
            } else {
                cur_scope_var_decls.push((Self::FOR_KEY_TEMP_VAR, token.byte_pos_start));
            }

            // This doesn't even need to be created if it's not set or underscore.
            if let Some(item_node) = item {
                add_decl_node!(item_node);
            }
        }

        for node in nodes {
            if matches!(&node.kind, AstNodeKind::DeclareAssign(_, _)) {
                add_decl_node!(node);
            }
        }

        let frame_ptr = self.cur_stack_ptr_offset;

        let add_sp = cur_scope_var_decls.len() as u32;
        if add_sp > 0 {
            self.push_instruction(Inst::add_stack_pointer(add_sp), node);
            self.cur_stack_ptr_offset += add_sp;
        }
        self.scope_var_decls.push(cur_scope_var_decls);

        let mut ctx = prev_ctx.clone();
        ctx.block_frame_ptr = frame_ptr;

        if fn_node.is_some() {
            ctx.fn_frame_ptr = frame_ptr;
        }

        if loop_node.is_some() {
            ctx.loop_frame_ptr = frame_ptr;
            ctx.loop_stack_ptr = self.cur_stack_ptr_offset;
        }

        ctx
    }

    fn block_end(&mut self, node: &'a AstNode, ctx: &Context) {
        let sub_sp = self.cur_stack_ptr_offset - ctx.block_frame_ptr;
        if sub_sp > 0 {
            self.push_instruction(Inst::sub_stack_pointer(sub_sp), node);
            self.cur_stack_ptr_offset -= sub_sp;
        }
        self.scope_var_decls.pop();
    }

    fn compile_block_full(&mut self, block: &'a AstNode, prev_ctx: &Context) {
        let mut ctx = self.block_start(block, prev_ctx, None, None);
        self.compile_block(block, &mut ctx);
        self.block_end(block, &ctx);
    }

    fn compile_block(&mut self, block: &'a AstNode, ctx: &mut Context) {
        let AstNodeKind::Block(b) = &block.kind else {
            self.fatal("Expected block node", block);
        };
        for node in b.iter() {
            match &node.kind {
                AstNodeKind::DeclareAssign(..) | AstNodeKind::Assign(..) => {
                    self.compile_assignment(node, ctx)
                }
                AstNodeKind::FunctionDefinition(..) => self.compile_function_definition(node, ctx),
                AstNodeKind::FunctionCall(..) => self.compile_function_call(RESULT_REG1, node, ctx),
                AstNodeKind::Return(..) => self.compile_return(node, ctx),
                AstNodeKind::ForLoop(..) => self.compile_for_loop(node, ctx),
                AstNodeKind::Continue => self.compile_continue(node, ctx),
                AstNodeKind::Break => self.compile_break(node, ctx),
                AstNodeKind::IfStatement(..) => self.compile_if_statement(node, ctx),
                _ => {
                    self.fatal("Unsupported AST node in compilation", node);
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Context {
    block_frame_ptr: u32,
    fn_frame_ptr: u32,
    loop_frame_ptr: u32,
    loop_stack_ptr: u32,
    loop_breaks: Vec<u32>,
    loop_continues: Vec<u32>,
}

#[allow(dead_code)]
pub fn compile<'a>(block: &'a AstNode, tokens: &'a [Token]) -> Compilation<'a> {
    let mut c = Compilation::new(tokens);
    let mut ctx = c.block_start(block, &Context::default(), None, None);
    c.compile_block(block, &mut ctx);
    c.block_end(block, &ctx);
    for (i, ins) in c.instructions.iter().enumerate() {
        trace!("{:4}: {}", i, ins);
    }
    c
}

pub fn binary_op_err(err: OpError, left_val: &Value, op: Operator, right_val: &Value) -> String {
    match err {
        OpError::InvalidOperandTypes => format!(
            "Cannot apply operator '{}' to operands {} and {})",
            op.dbg_display(),
            left_val.dbg_display(),
            right_val.dbg_display()
        ),
        OpError::DivisionByZero => "Division by zero".to_string(),
    }
}

// #[inline]
// fn promoted(l: &OldValue, r: &OldValue) -> Option<(f64, f64)> {
//     l.float_promoted().zip(r.float_promoted())
// }
//
// macro_rules! impl_op {
//     ($func_name:ident, $op:tt) => {
//         #[inline]
//         pub fn $func_name(err_fn: impl FnOnce(&str), l: &Value, r: &Value) -> Value {
//             match (l, r) {
//                 (Value::Integer(l), Value::Integer(r)) => (l $op r).into(),
//                 (Value::Float(l), Value::Float(r)) => (l $op r).into(),
//                 _ => promoted(l, r).map(|(l, r)| (l $op r).into()).unwrap_or_else(|| {
//                     err_fn(&binary_op_err(l, stringify!($op), r));
//                     unreachable!()
//                 }),
//             }
//         }
//     };
//     ($func_name:ident, $op:tt, $str_fn:expr) => {
//         #[inline]
//         pub fn $func_name(err_fn: impl FnOnce(&str), l: &Value, r: &Value) -> Value {
//             match (l, r) {
//                 (Value::Integer(l), Value::Integer(r)) => (l $op r).into(),
//                 (Value::Float(l), Value::Float(r)) => (l $op r).into(),
//                 (Value::String(l), Value::String(r)) => $str_fn(l, r).into(),
//                 _ => promoted(l, r).map(|(l, r)| (l $op r).into()).unwrap_or_else(|| {
//                     err_fn(&binary_op_err(l, stringify!($op), r));
//                     unreachable!()
//                 }),
//             }
//         }
//     };
// }
//
// impl_op!(add_op, +, |l: &Rc<String>, r: &Rc<String>| l.as_ref().to_owned() + r.as_ref());
// impl_op!(sub_op, -);
// impl_op!(mul_op, *);
// impl_op!(div_op, /);
// impl_op!(lt_op, <);
// impl_op!(gt_op, >);
// impl_op!(lte_op, <=);
// impl_op!(gte_op, >=);
// impl_op!(eq_op, ==, |l, r| l == r);
// impl_op!(neq_op, !=, |l, r| l != r);

#[inline]
pub fn binary_op(l: &Value, op: Operator, r: &Value) -> OpResult {
    match op {
        Operator::Add => l.add(r),
        Operator::Sub => l.sub(r),
        Operator::Mul => l.mul(r),
        Operator::Div => l.div(r),
        Operator::Lt => l.lt(r),
        Operator::Gt => l.gt(r),
        Operator::Lte => l.lte(r),
        Operator::Gte => l.gte(r),
        Operator::Eq => l.eq(r),
        Operator::Neq => l.neq(r),
    }
}

#[inline]
pub fn is_zero(cond_value: &Value) -> Option<bool> {
    Some(cond_value.as_int()? == 0)
}
