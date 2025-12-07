use std::rc::Rc;

use foldhash::{HashMap, HashMapExt};
use log::trace;

use crate::{
    ast_parser::{AstNode, AstNodeKind, FN_CALL_RETURN_ADDR_VAR, fatal_generic},
    builtins::{ArgsRequred, ProgramFn, all_builtins},
    tokenizer::{Operator, Token, Value},
};

/// This is very likely slow, it should just be a different instruction to use the stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Addr {
    Abs(u32),
    Stack(u32),
}

#[derive(Debug, Clone)]
pub struct BinaryOpData {
    pub dst: Addr,
    pub src1: Addr,
    pub src2: Addr,
}

#[derive(Debug, Clone)]
pub enum Inst {
    LoadAddr {
        dst: Addr,
        src: Addr,
    },
    LoadInt {
        dst: Addr,
        value: i64,
    },
    InitMapIterationList {
        dst: Addr,
    },
    LoadIterationKey {
        dst: Addr,
        src: Addr,
        index: Addr,
    },
    LoadCollectionItem {
        dst: Addr,
        src: Addr,
        key: Addr,
    },
    AddStackPointer {
        value: u32,
    },
    SubStackPointer {
        value: u32,
    },
    Add(BinaryOpData),
    Sub(BinaryOpData),
    Mul(BinaryOpData),
    Div(BinaryOpData),
    Lt(BinaryOpData),
    Gt(BinaryOpData),
    Lte(BinaryOpData),
    Gte(BinaryOpData),
    Eq(BinaryOpData),
    Neq(BinaryOpData),
    Incr {
        dst: Addr,
    },
    CallBuiltin {
        dst: Addr,
        func: usize,
        arg_count: u8,
    },
    Jump {
        target: usize,
    },
    JumpAddr {
        target: Addr,
    },
    JumpIfZero {
        target: usize,
        cond: Addr,
    },
}

impl Inst {
    fn set_incr_dst(&mut self, dst: Addr) {
        match self {
            Inst::Incr { dst: d } => *d = dst,
            _ => panic!("Cannot set incr dst on non-incr instruction"),
        }
    }

    fn set_jump_target(&mut self, target_ip: usize) {
        match self {
            Inst::Jump { target } => *target = target_ip,
            Inst::JumpIfZero { target, .. } => *target = target_ip,
            _ => panic!("Cannot set jump target on non-jump instruction"),
        }
    }
}

pub const ARG_REG_START: u32 = 0;
pub const ARG_REG_COUNT: u32 = 2 << 5;
const fn reg(n: u32) -> Addr {
    Addr::Abs(ARG_REG_START + ARG_REG_COUNT + n)
}
pub const RESULT_REG1: Addr = reg(0);
pub const RESULT_REG2: Addr = reg(1);
pub const ZERO_REG: Addr = reg(2);
pub const SUCCESS_FLAG_REG: Addr = reg(3);
pub const PLACEHOLDER_REG: Addr = reg(4);
pub const FN_CALL_RETURN_ADDR_REG: Addr = reg(5);
pub const FN_RETURN_VALUE_REG: Addr = reg(6);
pub const RESERVED_REGS: u32 = ARG_REG_START + ARG_REG_COUNT + 10;
pub const STACK_SIZE: u32 = 2 << 12;

pub struct Compilation<'a> {
    pub instructions: Vec<Inst>,
    pub literals: Vec<(Value, Addr)>,
    pub builtins: HashMap<String, (ProgramFn, usize, ArgsRequred)>,
    pub functions: HashMap<String, (usize, ArgsRequred)>,
    pub tokens: &'a [Token],
    pub ip_to_token: Vec<usize>,
    pub scope_vars: Vec<Vec<&'a str>>,
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
            scope_vars: Vec::new(),
            cur_stack_ptr_offset: 0,
        }
    }

    fn fatal(&self, msg: &str, node: &AstNode) -> ! {
        let token = &self.tokens[node.token_idx];
        fatal_generic(msg, "Fatal error during compilation", token)
    }

    fn cur_inst_ptr(&self) -> usize {
        self.instructions.len()
    }

    fn variable_offset(
        &mut self,
        name: &str,
        node: &AstNode,
        ctx: &mut BlockCtx,
        can_init: bool,
    ) -> Addr {
        trace!("{:?}", self.scope_vars);
        let mut frame_ptr = self.cur_stack_ptr_offset;
        let mut current_scope = true;
        let mut tried_to_use = false;
        for vars in self.scope_vars.iter().rev() {
            frame_ptr -= vars.len() as u32;
            if let Some(pos) = vars.iter().position(|v| *v == name) {
                // Initialize this variable if we are allowed and it is the next one to initialize
                // in this scope.
                if current_scope {
                    if can_init && pos as u32 == ctx.initialized_vars {
                        ctx.initialized_vars = pos as u32 + 1;
                    }
                    // If the variable is not the next to initialize, we are trying to access it too
                    // soon, but it could still exist in an outer scope.
                    else if pos as u32 >= ctx.initialized_vars {
                        tried_to_use = true;
                        current_scope = false;
                        continue;
                    }
                }
                let offset = self.cur_stack_ptr_offset - frame_ptr - pos as u32 - 1;
                trace!("variable offset: {} for variable '{}'", offset, name);
                return Addr::Stack(offset);
            }
            current_scope = false;
        }

        if tried_to_use {
            self.fatal(
                &format!("Variable '{}' used before initialization", name),
                node,
            );
        }
        self.fatal(&format!("Undefined variable: {}", name), node);
    }

    fn make_literal(&mut self, value: &Value) -> Addr {
        let addr = Addr::Abs(self.literals.len() as u32 + RESERVED_REGS);
        self.literals.push((value.clone(), addr));
        addr
    }

    fn push_instruction(&mut self, inst: Inst, node: &AstNode) {
        self.instructions.push(inst);
        self.ip_to_token.push(node.token_idx);
    }

    fn compile_assignment(&mut self, node: &AstNode, ctx: &mut BlockCtx) {
        let AstNodeKind::Assign(var, expr) = &node.kind else {
            unreachable!();
        };
        let var_addr = self.variable_offset(var, node, ctx, true);
        let (expr_addr, _) = self.compile_expression(expr, var_addr, ctx);
        if expr_addr != var_addr {
            self.push_instruction(
                Inst::LoadAddr {
                    dst: var_addr,
                    src: expr_addr,
                },
                node,
            );
        }
    }

    fn compile_function_definition(&mut self, node: &'a AstNode) {
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
        // Placeholder
        self.push_instruction(Inst::Jump { target: 0 }, node);

        let fn_start_ip = self.cur_inst_ptr();
        let mut ctx = self.block_start(body);

        let return_addr_var_addr =
            self.variable_offset(FN_CALL_RETURN_ADDR_VAR, node, &mut ctx, true);
        self.push_instruction(
            Inst::LoadAddr {
                dst: return_addr_var_addr,
                src: FN_CALL_RETURN_ADDR_REG,
            },
            node,
        );

        for (arg_idx, arg_name) in args.iter().enumerate() {
            let arg_addr = self.variable_offset(arg_name, node, &mut ctx, true);
            let arg_reg = Addr::Abs(ARG_REG_START + arg_idx as u32);
            self.push_instruction(
                Inst::LoadAddr {
                    dst: arg_addr,
                    src: arg_reg,
                },
                node,
            );
        }

        self.compile_block(body, &mut ctx, &mut LoopCtx::default());

        self.push_instruction(
            Inst::LoadInt {
                dst: FN_RETURN_VALUE_REG,
                value: 1,
            },
            node,
        );

        self.block_end(body, ctx);

        self.push_instruction(
            Inst::JumpAddr {
                target: return_addr_var_addr,
            },
            node,
        );

        let fn_end_ip = self.cur_inst_ptr();
        self.instructions[fn_skip_jump_ip].set_jump_target(fn_end_ip);

        let args_required = ArgsRequred::Exact(args.len() as u32);

        self.functions
            .insert(name.clone(), (fn_start_ip, args_required));
    }

    fn compile_set_function_args(&mut self, args: &[AstNode], ctx: &mut BlockCtx) {
        if args.len() > ARG_REG_COUNT as usize {
            self.fatal("Too many arguments in function call", &args[0]);
        }
        for (i, arg) in args.iter().enumerate() {
            let arg_reg = Addr::Abs(ARG_REG_START + i as u32);
            let (res_addr, _) = self.compile_expression(arg, arg_reg, ctx);
            if res_addr != arg_reg {
                self.push_instruction(
                    Inst::LoadAddr {
                        dst: arg_reg,
                        src: res_addr,
                    },
                    arg,
                );
            }
        }
    }

    fn compile_function_call(&mut self, dst: Addr, node: &AstNode, ctx: &mut BlockCtx) {
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

        self.compile_set_function_args(args, ctx);

        if let Some((_, func_index, args_req)) = self.builtins.get(name) {
            if !args_req.matches(args.len()) {
                unexpected_args!(args_req);
            }

            self.push_instruction(
                Inst::CallBuiltin {
                    dst,
                    func: *func_index,
                    arg_count: args.len() as u8,
                },
                node,
            );
        } else if let Some(&(fn_start_ip, args_req)) = self.functions.get(name) {
            if !args_req.matches(args.len()) {
                unexpected_args!(args_req);
            }

            let after_call_ip = self.cur_inst_ptr() + 2;
            self.push_instruction(
                Inst::LoadInt {
                    dst: FN_CALL_RETURN_ADDR_REG,
                    value: (after_call_ip as i64),
                },
                node,
            );
            self.push_instruction(
                Inst::Jump {
                    target: fn_start_ip,
                },
                node,
            );
            self.push_instruction(
                Inst::LoadAddr {
                    dst,
                    src: FN_RETURN_VALUE_REG,
                },
                node,
            );
        } else {
            self.fatal(&format!("Undefined function: {}", name), node);
        }
    }

    fn compile_expression(
        &mut self,
        expr: &AstNode,
        dst_suggestion: Addr,
        ctx: &mut BlockCtx,
    ) -> (Addr, Option<Value>) {
        match &expr.kind {
            AstNodeKind::Literal(value) => (self.make_literal(value), Some(value.clone())),
            AstNodeKind::Variable(name) => (self.variable_offset(name, expr, ctx, false), None),
            AstNodeKind::BinaryOp(left, op, right) => {
                let (laddr, lval) = self.compile_expression(left, RESULT_REG1, ctx);
                let (raddr, rval) = self.compile_expression(right, RESULT_REG2, ctx);

                // Constant folding for literals
                if let (Some(lit_left), Some(lit_right)) = (lval, rval) {
                    let folded_value =
                        binary_op(|err| self.fatal(err, expr), &lit_left, *op, &lit_right);
                    return (self.make_literal(&folded_value), Some(folded_value));
                }

                let binop_data = BinaryOpData {
                    dst: dst_suggestion,
                    src1: laddr,
                    src2: raddr,
                };
                let inst = match op {
                    Operator::Add => Inst::Add(binop_data),
                    Operator::Sub => Inst::Sub(binop_data),
                    Operator::Mul => Inst::Mul(binop_data),
                    Operator::Div => Inst::Div(binop_data),
                    Operator::Lt => Inst::Lt(binop_data),
                    Operator::Gt => Inst::Gt(binop_data),
                    Operator::Lte => Inst::Lte(binop_data),
                    Operator::Gte => Inst::Gte(binop_data),
                    Operator::Eq => Inst::Eq(binop_data),
                    Operator::Neq => Inst::Neq(binop_data),
                };

                self.push_instruction(inst, expr);
                (dst_suggestion, None)
            }
            AstNodeKind::FunctionCall(..) => {
                self.compile_function_call(dst_suggestion, expr, ctx);
                (dst_suggestion, None)
            }
            _ => todo!(),
        }
    }

    fn compile_for_loop(&mut self, node: &'a AstNode) {
        let AstNodeKind::ForLoop(index_var, key_var, item_var, collection, body) = &node.kind
        else {
            unreachable!();
        };
        let mut for_ctx = self.block_start(body);
        let mut loop_ctx = LoopCtx::new(for_ctx.frame_ptr, self.cur_stack_ptr_offset);

        let iterable_addr = self.variable_offset(
            self.scope_vars.last().unwrap().first().unwrap(),
            node,
            &mut for_ctx,
            true,
        );

        let (addr, _) = self.compile_expression(collection, iterable_addr, &mut for_ctx);
        if addr != iterable_addr {
            self.push_instruction(
                Inst::LoadAddr {
                    dst: iterable_addr,
                    src: addr,
                },
                collection,
            );
        };

        self.push_instruction(Inst::InitMapIterationList { dst: iterable_addr }, node);

        let index_addr = self.variable_offset(index_var, node, &mut for_ctx, true);
        self.push_instruction(
            Inst::LoadAddr {
                dst: index_addr,
                src: ZERO_REG,
            },
            node,
        );

        let for_load_key_ip = self.cur_inst_ptr();

        let key_addr = self.variable_offset(key_var, node, &mut for_ctx, true);
        self.push_instruction(
            Inst::LoadIterationKey {
                dst: key_addr,
                src: iterable_addr,
                index: index_addr,
            },
            node,
        );

        let for_exit_jump_ip = self.cur_inst_ptr();
        // Placeholder
        self.push_instruction(
            Inst::JumpIfZero {
                target: 0,
                cond: SUCCESS_FLAG_REG,
            },
            node,
        );

        if let Some(item_var) = item_var {
            let item_addr = self.variable_offset(item_var, node, &mut for_ctx, true);
            self.push_instruction(
                Inst::LoadCollectionItem {
                    dst: item_addr,
                    src: iterable_addr,
                    key: key_addr,
                },
                node,
            );
        } else {
            for_ctx.initialized_vars += 1;
        }

        self.compile_block(body, &mut for_ctx, &mut loop_ctx);

        self.push_instruction(Inst::Incr { dst: index_addr }, node);
        self.push_instruction(
            Inst::Jump {
                target: for_load_key_ip,
            },
            node,
        );

        let loop_end_ip = self.cur_inst_ptr();
        self.instructions[for_exit_jump_ip].set_jump_target(loop_end_ip);

        for continue_ip in &loop_ctx.continues {
            self.instructions[*continue_ip].set_incr_dst(index_addr);
            self.instructions[*continue_ip + 1].set_jump_target(for_load_key_ip);
        }

        self.block_end(body, for_ctx);

        let loop_end_after_sp_reset_ip = self.cur_inst_ptr();

        for break_ip in &loop_ctx.breaks {
            self.instructions[*break_ip].set_jump_target(loop_end_after_sp_reset_ip);
        }
    }

    fn compile_continue(&mut self, node: &'a AstNode, loop_ctx: &mut LoopCtx) {
        let sub_sp = self.cur_stack_ptr_offset - loop_ctx.stack_ptr;
        if sub_sp > 0 {
            self.push_instruction(Inst::SubStackPointer { value: sub_sp }, node);
        }
        loop_ctx.continues.push(self.cur_inst_ptr());
        // Placeholder
        self.push_instruction(
            Inst::Incr {
                dst: PLACEHOLDER_REG,
            },
            node,
        );
        // Placeholder
        self.push_instruction(Inst::Jump { target: 0 }, node);
    }

    fn compile_break(&mut self, node: &'a AstNode, loop_ctx: &mut LoopCtx) {
        let sub_sp = self.cur_stack_ptr_offset - loop_ctx.frame_ptr;
        if sub_sp > 0 {
            self.push_instruction(Inst::SubStackPointer { value: sub_sp }, node);
        }
        loop_ctx.breaks.push(self.cur_inst_ptr());
        // Placeholder
        self.push_instruction(Inst::Jump { target: 0 }, node);
    }

    fn compile_if_statement(
        &mut self,
        node: &'a AstNode,
        ctx: &mut BlockCtx,
        loop_ctx: &mut LoopCtx,
    ) {
        let AstNodeKind::IfStatement(condition, block, else_block) = &node.kind else {
            unreachable!();
        };
        let (cond_addr, cond_val) = self.compile_expression(condition, RESULT_REG1, ctx);

        let const_cond_true = cond_val.map(|v| !is_zero(|s| self.fatal(s, node), &v));

        if let Some(const_cond_true) = const_cond_true {
            if const_cond_true {
                self.compile_block_full(block, loop_ctx);
            } else if let Some(else_block) = else_block {
                self.compile_block_full(else_block, loop_ctx);
            }
            return;
        }

        let if_jump_ip = self.cur_inst_ptr();
        // Placeholder
        self.push_instruction(
            Inst::JumpIfZero {
                target: 0,
                cond: cond_addr,
            },
            node,
        );

        self.compile_block_full(block, loop_ctx);

        if let Some(else_block) = else_block {
            let else_jump_ip = self.cur_inst_ptr();
            // Placeholder
            self.push_instruction(Inst::Jump { target: 0 }, node);

            let else_start_ip = self.cur_inst_ptr();
            self.instructions[if_jump_ip].set_jump_target(else_start_ip);

            self.compile_block_full(else_block, loop_ctx);

            let after_else_ip = self.cur_inst_ptr();
            self.instructions[else_jump_ip].set_jump_target(after_else_ip);
        } else {
            let after_if_ip = self.cur_inst_ptr();
            self.instructions[if_jump_ip].set_jump_target(after_if_ip);
        }
    }

    fn block_start(&mut self, node: &'a AstNode) -> BlockCtx {
        let AstNodeKind::Block(_, vars) = &node.kind else {
            self.fatal("Expected block node", node);
        };
        let mut cur_scope_vars: Vec<&str> = Vec::with_capacity(vars.len());
        for var in vars {
            if !self.scope_vars.iter().flatten().any(|v| v == var) {
                cur_scope_vars.push(var);
            }
        }

        let frame_ptr = self.cur_stack_ptr_offset;

        let add_sp = cur_scope_vars.len() as u32;
        if add_sp > 0 {
            self.push_instruction(Inst::AddStackPointer { value: add_sp }, node);
            self.cur_stack_ptr_offset += add_sp;
        }
        self.scope_vars.push(cur_scope_vars);
        BlockCtx::new(frame_ptr)
    }

    fn block_end(&mut self, node: &'a AstNode, ctx: BlockCtx) {
        let sub_sp = self.cur_stack_ptr_offset - ctx.frame_ptr;
        if sub_sp > 0 {
            self.push_instruction(Inst::SubStackPointer { value: sub_sp }, node);
            self.cur_stack_ptr_offset -= sub_sp;
        }
        self.scope_vars.pop();
    }

    fn compile_block_full(&mut self, block: &'a AstNode, loop_ctx: &mut LoopCtx) {
        let mut ctx = self.block_start(block);
        self.compile_block(block, &mut ctx, loop_ctx);
        self.block_end(block, ctx);
    }

    fn compile_block(&mut self, block: &'a AstNode, ctx: &mut BlockCtx, loop_ctx: &mut LoopCtx) {
        let AstNodeKind::Block(b, _) = &block.kind else {
            self.fatal("Expected block node", block);
        };
        for node in b.iter() {
            match &node.kind {
                AstNodeKind::Assign(..) => self.compile_assignment(node, ctx),
                AstNodeKind::FunctionDefinition(..) => self.compile_function_definition(node),
                AstNodeKind::FunctionCall(..) => self.compile_function_call(RESULT_REG1, node, ctx),
                AstNodeKind::ForLoop(..) => self.compile_for_loop(node),
                AstNodeKind::IfStatement(..) => self.compile_if_statement(node, ctx, loop_ctx),
                AstNodeKind::Continue => self.compile_continue(node, loop_ctx),
                AstNodeKind::Break => self.compile_break(node, loop_ctx),
                _ => {
                    self.fatal("Unsupported AST node in compilation", node);
                }
            }
        }
    }
}

struct BlockCtx {
    frame_ptr: u32,
    initialized_vars: u32,
}

impl BlockCtx {
    fn new(frame_ptr: u32) -> Self {
        BlockCtx {
            frame_ptr,
            initialized_vars: 0,
        }
    }
}

#[derive(Default)]
struct LoopCtx {
    frame_ptr: u32,
    stack_ptr: u32,
    breaks: Vec<usize>,
    continues: Vec<usize>,
}

impl LoopCtx {
    fn new(frame_ptr: u32, stack_ptr: u32) -> Self {
        LoopCtx {
            frame_ptr,
            stack_ptr,
            breaks: Vec::new(),
            continues: Vec::new(),
        }
    }
}

#[allow(dead_code)]
pub fn compile<'a>(block: &'a AstNode, tokens: &'a [Token]) -> Compilation<'a> {
    let mut c = Compilation::new(tokens);
    let mut ctx = c.block_start(block);
    let mut loop_ctx = LoopCtx::default();
    c.compile_block(block, &mut ctx, &mut loop_ctx);
    c.block_end(block, ctx);
    for (i, ins) in c.instructions.iter().enumerate() {
        trace!("{:4}: {:?}", i, ins);
    }
    c
}

pub fn binary_op_err(left_val: &Value, op: &str, right_val: &Value) -> String {
    format!(
        "Cannot apply operator '{}' to operands {} and {})",
        op,
        left_val.dbg_display(),
        right_val.dbg_display()
    )
}

#[inline]
fn promoted(l: &Value, r: &Value) -> Option<(f64, f64)> {
    l.float_promoted().zip(r.float_promoted())
}

macro_rules! impl_op {
    ($func_name:ident, $op:tt) => {
        #[inline]
        pub fn $func_name(err_fn: impl FnOnce(&str), l: &Value, r: &Value) -> Value {
            match (l, r) {
                (Value::Integer(l), Value::Integer(r)) => (l $op r).into(),
                (Value::Float(l), Value::Float(r)) => (l $op r).into(),
                _ => promoted(l, r).map(|(l, r)| (l $op r).into()).unwrap_or_else(|| {
                    err_fn(&binary_op_err(l, stringify!($op), r));
                    unreachable!()
                }),
            }
        }
    };
    ($func_name:ident, $op:tt, $str_fn:expr) => {
        #[inline]
        pub fn $func_name(err_fn: impl FnOnce(&str), l: &Value, r: &Value) -> Value {
            match (l, r) {
                (Value::Integer(l), Value::Integer(r)) => (l $op r).into(),
                (Value::Float(l), Value::Float(r)) => (l $op r).into(),
                (Value::String(l), Value::String(r)) => $str_fn(l, r).into(),
                _ => promoted(l, r).map(|(l, r)| (l $op r).into()).unwrap_or_else(|| {
                    err_fn(&binary_op_err(l, stringify!($op), r));
                    unreachable!()
                }),
            }
        }
    };
}

impl_op!(add_op, +, |l: &Rc<String>, r: &Rc<String>| l.as_ref().to_owned() + r.as_ref());
impl_op!(sub_op, -);
impl_op!(mul_op, *);
impl_op!(div_op, /);
impl_op!(lt_op, <);
impl_op!(gt_op, >);
impl_op!(lte_op, <=);
impl_op!(gte_op, >=);
impl_op!(eq_op, ==, |l, r| l == r);
impl_op!(neq_op, !=, |l, r| l != r);

#[inline]
pub fn binary_op(err_fn: impl FnOnce(&str), l: &Value, op: Operator, r: &Value) -> Value {
    match op {
        Operator::Add => add_op(err_fn, l, r),
        Operator::Sub => sub_op(err_fn, l, r),
        Operator::Mul => mul_op(err_fn, l, r),
        Operator::Div => div_op(err_fn, l, r),
        Operator::Lt => lt_op(err_fn, l, r),
        Operator::Gt => gt_op(err_fn, l, r),
        Operator::Lte => lte_op(err_fn, l, r),
        Operator::Gte => gte_op(err_fn, l, r),
        Operator::Eq => eq_op(err_fn, l, r),
        Operator::Neq => neq_op(err_fn, l, r),
    }
}

#[inline]
pub fn is_zero(err_fn: impl FnOnce(&str), cond_value: &Value) -> bool {
    match cond_value {
        Value::Integer(i) => *i == 0,
        _ => {
            err_fn(&format!(
                "Expected (int) as condition, got {:?}",
                cond_value.dbg_display()
            ));
            unreachable!()
        }
    }
}
