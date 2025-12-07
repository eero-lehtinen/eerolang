use std::rc::Rc;

use foldhash::{HashMap, HashMapExt};
use log::trace;

use crate::{
    ast_parser::{AstNode, AstNodeKind, fatal_generic},
    builtins::{ProgramFn, all_builtins},
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
    Load {
        dst: Addr,
        src: Addr,
    },
    LoadFromCollection {
        dst: Addr,
        collection: Addr,
        index: Addr,
    },
    PushToCollection {
        collection: Addr,
        value: Addr,
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
    JumpIfNotInRange {
        target: usize,
        range: Addr,
        src: Addr,
    },
    Call {
        dst: Addr,
        func: usize,
        arg_count: u8,
    },
    Jump {
        target: usize,
    },
    JumpIfZero {
        target: usize,
        cond: Addr,
    },
}

impl Inst {
    fn set_jump_target(&mut self, target_ip: usize) {
        match self {
            Inst::Jump { target } => *target = target_ip,
            Inst::JumpIfZero { target, .. } => *target = target_ip,
            Inst::JumpIfNotInRange { target, .. } => *target = target_ip,
            _ => panic!("Cannot set jump target on non-jump instruction"),
        }
    }
}

pub const RESULT_REG1: Addr = Addr::Abs(0);
pub const RESULT_REG2: Addr = Addr::Abs(1);
pub const ZERO_REG: Addr = Addr::Abs(2);
pub const EMPTY_LIST_REG: Addr = Addr::Abs(4);
pub const ARG_REG_START: u32 = 10;
pub const ARG_REG_COUNT: u32 = 10;
pub const RESERVED_REGS: u32 = 30;
pub const STACK_SIZE: u32 = 2 << 11;

pub struct Compilation<'a> {
    pub instructions: Vec<Inst>,
    pub literals: Vec<(Value, Addr)>,
    pub functions: HashMap<String, (ProgramFn, usize)>,
    pub tokens: &'a [Token],
    pub ip_to_token: Vec<usize>,
    pub scope_vars: Vec<Vec<&'a str>>,
    cur_stack_ptr_offset: u32,
}

impl<'a> Compilation<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        let mut functions = HashMap::new();
        for (i, (name, func)) in all_builtins().iter().enumerate() {
            functions.insert(name.to_string(), (*func, i));
        }
        Compilation {
            instructions: Vec::new(),
            literals: Vec::new(),
            functions,
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
        initialized_vars: &mut u32,
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
                    if can_init && pos as u32 == *initialized_vars {
                        *initialized_vars = pos as u32 + 1;
                    }
                    // If the variable is not the next to initialize, we are trying to access it too
                    // soon, but it could still exist in an outer scope.
                    else if pos as u32 >= *initialized_vars {
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
        match &inst {
            Inst::AddStackPointer { value } => {
                self.cur_stack_ptr_offset += *value;
            }
            Inst::SubStackPointer { value } => {
                self.cur_stack_ptr_offset -= *value;
            }
            _ => (),
        }
        self.instructions.push(inst);
        self.ip_to_token.push(node.token_idx);
    }

    fn compile_function_args(&mut self, args: &[AstNode], initialized_vars: &mut u32) {
        if args.len() > ARG_REG_COUNT as usize {
            self.fatal("Too many arguments in function call", &args[0]);
        }
        for (i, arg) in args.iter().enumerate() {
            let arg_reg = Addr::Abs(ARG_REG_START + i as u32);
            let (res_reg, _) = self.compile_expression(arg, arg_reg, initialized_vars);
            if res_reg != arg_reg {
                self.push_instruction(
                    Inst::Load {
                        dst: arg_reg,
                        src: res_reg,
                    },
                    arg,
                );
            }
        }
    }

    #[allow(dead_code)]
    fn compile_function_args_internal(&mut self, args: &[u32]) {
        if args.len() > ARG_REG_COUNT as usize {
            panic!("Too many arguments in function call");
        }
        for (i, &arg_reg) in args.iter().enumerate() {
            let target_reg = ARG_REG_START + i as u32;
            self.push_instruction(
                Inst::Load {
                    dst: Addr::Abs(target_reg),
                    src: Addr::Abs(arg_reg),
                },
                &AstNode {
                    kind: AstNodeKind::Literal(Value::Integer(0)),
                    token_idx: 0,
                },
            );
        }
    }

    fn compile_function_call(
        &mut self,
        name: &str,
        arg_count: usize,
        dst: Addr,
        node: &AstNode,
    ) -> Addr {
        let (_, func_index) = self
            .functions
            .get(name)
            .unwrap_or_else(|| self.fatal(&format!("Undefined function: {}", name), node));

        self.push_instruction(
            Inst::Call {
                dst,
                func: *func_index,
                arg_count: arg_count as u8,
            },
            node,
        );

        dst
    }

    fn compile_expression(
        &mut self,
        expr: &AstNode,
        dst_suggestion: Addr,
        initialized_vars: &mut u32,
    ) -> (Addr, Option<Value>) {
        match &expr.kind {
            AstNodeKind::Literal(value) => (self.make_literal(value), Some(value.clone())),
            AstNodeKind::Variable(name) => (
                self.variable_offset(name, expr, initialized_vars, false),
                None,
            ),
            AstNodeKind::BinaryOp(left, op, right) => {
                let (lreg, lval) = self.compile_expression(left, RESULT_REG1, initialized_vars);
                let (rreg, rval) = self.compile_expression(right, RESULT_REG2, initialized_vars);

                // Constant folding for literals
                if let (Some(lit_left), Some(lit_right)) = (lval, rval) {
                    let folded_value =
                        binary_op(|err| self.fatal(err, expr), &lit_left, *op, &lit_right);
                    return (self.make_literal(&folded_value), Some(folded_value));
                }

                let binop_data = BinaryOpData {
                    dst: dst_suggestion,
                    src1: lreg,
                    src2: rreg,
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
            AstNodeKind::FunctionCall(name, args) => {
                self.compile_function_args(args, initialized_vars);
                let dst = self.compile_function_call(name, args.len(), dst_suggestion, expr);
                (dst, None)
            }
            AstNodeKind::List(nodes) => {
                self.push_instruction(
                    Inst::Load {
                        dst: dst_suggestion,
                        src: EMPTY_LIST_REG,
                    },
                    expr,
                );
                for node in nodes.iter() {
                    let (value_reg, _) =
                        self.compile_expression(node, RESULT_REG1, initialized_vars);
                    self.push_instruction(
                        Inst::PushToCollection {
                            collection: dst_suggestion,
                            value: value_reg,
                        },
                        node,
                    );
                }
                (dst_suggestion, None)
            }
            _ => todo!(),
        }
    }

    fn block_start(&mut self, node: &'a AstNode) -> u32 {
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
        self.push_instruction(
            Inst::AddStackPointer {
                value: cur_scope_vars.len() as u32,
            },
            node,
        );
        self.scope_vars.push(cur_scope_vars);
        frame_ptr
    }

    fn block_end(&mut self, frame_ptr: u32, node: &'a AstNode) {
        self.push_instruction(
            Inst::SubStackPointer {
                value: self.cur_stack_ptr_offset - frame_ptr,
            },
            node,
        );
        self.scope_vars.pop();
    }

    fn compile_block_full(&mut self, block: &'a AstNode) {
        let frame_ptr = self.block_start(block);
        let mut initialized_vars = 0;
        self.compile_block(block, &mut initialized_vars);
        self.block_end(frame_ptr, block);
    }

    fn compile_block(&mut self, block: &'a AstNode, initialized_vars: &mut u32) {
        let AstNodeKind::Block(b, _) = &block.kind else {
            self.fatal("Expected block node", block);
        };
        for node in b.iter() {
            match &node.kind {
                AstNodeKind::Assign(var, expr) => {
                    let var_reg = self.variable_offset(var, node, initialized_vars, true);
                    let (expr_reg, _) = self.compile_expression(expr, var_reg, initialized_vars);
                    if expr_reg != var_reg {
                        self.push_instruction(
                            Inst::Load {
                                dst: var_reg,
                                src: expr_reg,
                            },
                            node,
                        );
                    }
                }
                AstNodeKind::FunctionCall(name, args) => {
                    self.compile_function_args(args, initialized_vars);
                    self.compile_function_call(name, args.len(), RESULT_REG1, node);
                }
                AstNodeKind::ForLoop(index_var, item_var, collection, body) => {
                    let frame_ptr = self.block_start(body);
                    let mut initialized_vars = 0;

                    let iterable_addr = self.variable_offset(
                        self.scope_vars.last().unwrap().first().unwrap(),
                        node,
                        &mut initialized_vars,
                        true,
                    );

                    let (addr, _) =
                        self.compile_expression(collection, iterable_addr, &mut initialized_vars);
                    if addr != iterable_addr {
                        self.push_instruction(
                            Inst::Load {
                                dst: iterable_addr,
                                src: addr,
                            },
                            collection,
                        );
                    };

                    let index_addr =
                        self.variable_offset(index_var, node, &mut initialized_vars, true);
                    self.push_instruction(
                        Inst::Load {
                            dst: index_addr,
                            src: ZERO_REG,
                        },
                        node,
                    );

                    let for_cmp_ip = self.cur_inst_ptr();
                    self.push_instruction(
                        Inst::JumpIfNotInRange {
                            target: 0, // Placeholder, will be filled later
                            range: iterable_addr,
                            src: index_addr,
                        },
                        node,
                    );

                    if let Some(item_var) = item_var
                        && item_var != "_"
                    {
                        let item_addr =
                            self.variable_offset(item_var, node, &mut initialized_vars, true);
                        self.push_instruction(
                            Inst::LoadFromCollection {
                                dst: item_addr,
                                collection: iterable_addr,
                                index: index_addr,
                            },
                            node,
                        );
                    } else {
                        initialized_vars += 1;
                    }

                    self.compile_block(body, &mut initialized_vars);

                    self.push_instruction(Inst::Incr { dst: index_addr }, node);
                    self.push_instruction(Inst::Jump { target: for_cmp_ip }, node);

                    let loop_end_ip = self.cur_inst_ptr();
                    self.instructions[for_cmp_ip].set_jump_target(loop_end_ip);

                    self.block_end(frame_ptr, body);
                }
                AstNodeKind::IfStatement(condition, block, else_block) => {
                    let (cond_reg, cond_val) =
                        self.compile_expression(condition, RESULT_REG1, initialized_vars);

                    let const_cond_true = cond_val.map(|v| !is_zero(|s| self.fatal(s, node), &v));

                    if let Some(const_cond_true) = const_cond_true {
                        if const_cond_true {
                            self.compile_block_full(block);
                        } else if let Some(else_block) = else_block {
                            self.compile_block_full(else_block);
                        }
                        continue;
                    }

                    let if_jump_ip = self.cur_inst_ptr();
                    self.push_instruction(
                        Inst::JumpIfZero {
                            target: 0, // Placeholder
                            cond: cond_reg,
                        },
                        node,
                    );

                    self.compile_block_full(block);

                    if let Some(else_block) = else_block {
                        let else_jump_ip = self.cur_inst_ptr();
                        self.push_instruction(
                            Inst::Jump {
                                target: 0, // Placeholder
                            },
                            node,
                        );

                        let else_start_ip = self.cur_inst_ptr();
                        self.instructions[if_jump_ip].set_jump_target(else_start_ip);

                        self.compile_block_full(else_block);

                        let after_else_ip = self.cur_inst_ptr();
                        self.instructions[else_jump_ip].set_jump_target(after_else_ip);
                    } else {
                        let after_if_ip = self.cur_inst_ptr();
                        self.instructions[if_jump_ip].set_jump_target(after_if_ip);
                    }
                }
                _ => {
                    self.fatal("Unsupported AST node in compilation", node);
                }
            }
        }
    }
}

#[allow(dead_code)]
pub fn compile<'a>(block: &'a AstNode, tokens: &'a [Token]) -> Compilation<'a> {
    let mut c = Compilation::new(tokens);
    let mut initialized_vars = 0;
    let frame_ptr = c.block_start(block);
    c.compile_block(block, &mut initialized_vars);
    c.block_end(frame_ptr, block);
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
