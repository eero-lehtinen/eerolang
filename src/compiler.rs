use foldhash::{HashMap, HashMapExt};
use log::trace;

use crate::{
    ast_parser::{AstNode, AstNodeKind, fatal_generic},
    builtins::{ArgsRequred, ProgramFn, all_builtins},
    instructions::{Addr, ConstAddr, GlobalAddr, Inst},
    tokenizer::{Literal, Operator, Token},
    value::{OpResult, Value},
};

#[derive(Debug)]
struct ScopeData<'a> {
    declarations: Vec<&'a str>,
    /// Stored at the root of the scope stack for function scopes.
    fn_data: Option<FnData>,
    loop_data: Option<LoopData>,
}

#[derive(Debug, Default)]
struct FnData {
    /// Keeps track of variables declared in the function to figure out stack allocation for locals.
    /// Holds (name, scope depth)
    locals: Vec<(String, u32)>,
}

#[derive(Debug)]
struct LoopData {
    breaks: Vec<u32>,
    continues: Vec<u32>,
}

pub struct Compilation<'a> {
    pub instructions: Vec<Inst>,
    /// Holds literal values used in the program.
    pub constants: Vec<Value>,
    /// Keeps track of variables declared outside of functions to figure out how much space to
    /// allocate at the start for globals (globals can also be within loops and if blocks).
    /// Holds (name, scope_depth).
    pub globals: Vec<(String, u32)>,
    pub builtins: HashMap<String, (ProgramFn, usize, ArgsRequred)>,
    pub functions: HashMap<String, (u32, ArgsRequred)>,
    pub tokens: &'a [Token],
    pub ip_to_token: Vec<usize>,
    scopes: Vec<ScopeData<'a>>,
    cur_stack_ptr_offset: u32,
}

const ZERO_CONST_ADDR: ConstAddr = ConstAddr(0);
const NEG_ONE_ADDR: ConstAddr = ConstAddr(1);

impl<'a> Compilation<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        let mut builtins = HashMap::new();
        for (i, (name, func, args)) in all_builtins().iter().enumerate() {
            builtins.insert(name.to_string(), (*func, i, *args));
        }
        Compilation {
            instructions: Vec::new(),
            constants: vec![const { Value::smi(0) }, const { Value::smi(-1) }],
            globals: Vec::new(),
            builtins,
            functions: HashMap::new(),
            tokens,
            ip_to_token: Vec::new(),
            scopes: Vec::new(),
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

    fn declare_variable(&mut self, name: &'a str, node: &'a AstNode) -> Addr {
        if let Some(_) = self.fn_data() {
            todo!();
        } else {
            let depth = (self.scopes.len() - 1) as u32;

            if self.globals.iter().any(|(n, d)| n == name && *d == depth) {
                self.fatal(
                    &format!("Variable '{}' already declared in this scope", name),
                    node,
                );
            }

            self.scopes
                .last_mut()
                .expect("At least one scope exists")
                .declarations
                .push(name);

            let offset = self.globals.len() as u32;
            self.globals.push((name.to_string(), depth));
            Addr::Global(GlobalAddr(offset))
        }
    }

    fn variable_addr(&mut self, name: &str, node: &AstNode, to_decl: Option<&str>) -> Addr {
        trace!("{:?}", self.scopes);

        // If we are currently assigning to a declaration with the same name, we
        // should search for outer scopes to shadow properly.
        let skip_scope = if let Some(to_decl) = to_decl
            && to_decl == name
        {
            1
        } else {
            0
        };

        for (depth, scope) in self.scopes.iter().enumerate().rev().skip(skip_scope) {
            for decl_name in scope.declarations.iter() {
                if *decl_name == name {
                    let Some(idx) = self
                        .globals
                        .iter()
                        .position(|(n, d)| n == name && *d as usize == depth)
                    else {
                        self.fatal(&format!("Variable '{}' not declared", name), node);
                    };
                    return Addr::Global(GlobalAddr(idx as u32));
                }
            }
        }

        self.fatal(&format!("Variable '{}' not declared", name), node);
    }

    fn push_literal(&mut self, value: &Literal) -> ConstAddr {
        let value = match value {
            Literal::Number(n) => Value::number(*n),
            Literal::String(s) => Value::string(s.clone()),
        };
        let addr = self.constants.len() as u32;
        self.constants.push(value);
        ConstAddr(addr)
    }

    fn push_instruction(&mut self, inst: Inst, node: &AstNode) {
        self.instructions.push(inst);
        self.ip_to_token.push(node.token_idx);
    }

    fn compile_assignment(&mut self, node: &'a AstNode) {
        let (var, expr, decl) = match &node.kind {
            AstNodeKind::DeclareAssign(var, expr) => (var, expr, true),
            AstNodeKind::Assign(var, expr) => (var, expr, false),
            _ => unreachable!(),
        };
        let var_addr = if decl {
            self.declare_variable(var, node)
        } else {
            self.variable_addr(var, node, None)
        };
        self.compile_expression(expr, decl.then_some(var));
        match var_addr {
            Addr::Const(_) => self.fatal("Cannot assign to constant", node),
            Addr::Global(addr) => {
                self.push_instruction(Inst::StoreGlobal(addr), node);
            }
            Addr::Local(addr) => {
                self.push_instruction(Inst::StoreLocal(addr), node);
            }
        }
    }

    fn compile_expression(&mut self, expr: &AstNode, to_decl: Option<&str>) {
        match &expr.kind {
            AstNodeKind::Literal(literal) => {
                let addr = self.push_literal(literal);
                self.push_instruction(Inst::LoadConst(addr), expr);
            }
            AstNodeKind::Variable(name) => {
                let addr = self.variable_addr(name, expr, to_decl);
                match addr {
                    Addr::Const(_) => self.fatal("Cannot use constant as variable", expr),
                    Addr::Global(addr) => {
                        self.push_instruction(Inst::LoadGlobal(addr), expr);
                    }
                    Addr::Local(addr) => {
                        self.push_instruction(Inst::LoadLocal(addr), expr);
                    }
                }
            }
            AstNodeKind::BinaryOp(left, op, right) => {
                self.compile_expression(left, to_decl);
                if *op == Operator::And || *op == Operator::Or {
                    let short_circuit_ip = self.cur_inst_ptr();
                    self.push_instruction(Inst::Nop, expr);
                    self.compile_expression(right, to_decl);
                    let after_right_ip = self.cur_inst_ptr();
                    if *op == Operator::And {
                        *self.inst_mut(short_circuit_ip) = Inst::JumpIfFalsy(after_right_ip);
                    } else {
                        *self.inst_mut(short_circuit_ip) = Inst::JumpIfTruthy(after_right_ip);
                    }
                } else {
                    self.compile_expression(right, to_decl);
                    self.push_instruction(Inst::binary_op(*op), expr);
                }
            }
            AstNodeKind::FunctionCall(..) => {
                self.compile_function_call(expr, false);
            }
            _ => todo!(),
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
        // Function instructions are defined "in the middle" of the instructions so we need to skip
        // over it to make top level code work correctly.
        self.push_instruction(Inst::Nop, node);

        // let fn_start_ip = self.cur_inst_ptr();
        //
        // let args_required = ArgsRequred::Exact(args.len() as u32);
        // self.functions
        //     .insert(name.clone(), (fn_start_ip, args_required));
        //
        // self.block_start(body, Some(node), false);
        //
        // self.compile_block(body);
        //
        // // Default return value is 1
        // self.push_instruction(Inst::load_int(FN_RETURN_VALUE_REG, 1), node);
        //
        // // Clean up stack frame.
        // self.block_end(body);
        //
        // // Jump back to return address.
        // self.push_instruction(Inst::jump_addr(FN_CALL_RETURN_ADDR_REG), node);
        //
        // let fn_end_ip = self.cur_inst_ptr();
        // self.inst_mut(fn_skip_jump_ip).set_jump_target(fn_end_ip);
    }

    fn compile_return(&mut self, node: &'a AstNode) {
        todo!();
        // let AstNodeKind::Return(expr) = &node.kind else {
        //     unreachable!();
        // };
        // let (expr_addr, _) = self.compile_expression(expr, FN_RETURN_VALUE_REG, None);
        // if expr_addr != FN_RETURN_VALUE_REG {
        //     self.push_instruction(Inst::load_addr(FN_RETURN_VALUE_REG, expr_addr), node);
        // }
        //
        // // Clean up stack frame.
        // let sub_sp = self.cur_stack_ptr_offset - self.fn_data().frame_ptr;
        // if sub_sp > 0 {
        //     self.push_instruction(Inst::sub_stack_pointer(sub_sp), node);
        // }
        //
        // // Jump back to return address.
        // self.push_instruction(Inst::jump_addr(FN_CALL_RETURN_ADDR_REG), node);
    }

    fn compile_function_args(&mut self, args: &[AstNode]) {
        for arg in args {
            self.compile_expression(arg, None);
        }
    }

    fn compile_function_call(&mut self, node: &AstNode, discard_returned: bool) {
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

        self.compile_function_args(args);

        if let Some((_, func_index, args_req)) = self.builtins.get(name).cloned() {
            if !args_req.matches(args.len()) {
                unexpected_args!(args_req);
            }

            self.push_instruction(
                Inst::CallBuiltin(func_index as u32, args.len() as u32),
                node,
            );
        } else {
            todo!()
        }

        if discard_returned {
            // Discard return value
            self.push_instruction(Inst::Pop, node);
        }

        // } else if let Some(&(fn_start_ip, args_req)) = self.functions.get(name) {
        //     if !args_req.matches(args.len()) {
        //         unexpected_args!(args_req);
        //     }
        //
        //     // Store return address (placeholder)
        //     let load_ret_addr_ip = self.cur_inst_ptr();
        //     self.push_instruction(Inst::nop(), node);
        //
        //     // Jump to the function.
        //     self.push_instruction(Inst::jump(fn_start_ip), node);
        //
        //     // Store return address now that we know it.
        //     *self.inst_mut(load_ret_addr_ip) =
        //         Inst::load_int(FN_CALL_RETURN_ADDR_REG, self.cur_inst_ptr() as i32);
        // } else {
        //     self.fatal(&format!("Undefined function: {}", name), node);
        // }
        //
        // // Restore temporaries after the function call.
        // self.push_instruction(Inst::restore_regs(save_args_count), node);
        // self.cur_stack_ptr_offset -= save_regs_count + save_args_count;
        //
        // // Load return value to the correct location.
        // self.push_instruction(Inst::load_addr(dst, FN_RETURN_VALUE_REG), node);
    }

    // If the iterable is an expression, it needs to be stored somwhere.
    const FOR_ITERABLE_TEMP_VAR: &'static str = "__for_iterable_temp";
    // Index needs to be stored somewhere.
    const FOR_INDEX_TEMP_VAR: &'static str = "__for_index_temp";
    // Even if not assigned to a variable, the key needs to be stored somewhere.
    const FOR_KEY_TEMP_VAR: &'static str = "__for_key_temp";

    fn compile_loop(&mut self, node: &'a AstNode) {
        let (loop_next_iteration_ip, loop_continue_ip, loop_exit_jump_ip) =
            if let AstNodeKind::ForLoop(key, item, iterable, body) = &node.kind {
                self.block_start(body, None, true);

                let iterable_addr = self.declare_variable(Self::FOR_ITERABLE_TEMP_VAR, iterable);
                self.compile_expression(iterable, Some(Self::FOR_ITERABLE_TEMP_VAR));
                self.push_instruction(Inst::store(iterable_addr), iterable);
                self.push_instruction(Inst::load(iterable_addr), iterable);
                self.push_instruction(Inst::InitMapIter, iterable);

                let index_addr = self.declare_variable(Self::FOR_INDEX_TEMP_VAR, body);
                self.push_instruction(Inst::LoadConst(ZERO_CONST_ADDR), body);
                self.push_instruction(Inst::store(index_addr), body);

                let loop_next_iteration_ip = self.cur_inst_ptr();

                let (key_var_name, key_node) = if let Some(key_node) = key {
                    (
                        key_node.get_var_name().expect("Parsed correctly"),
                        key_node.as_ref(),
                    )
                } else {
                    (Self::FOR_KEY_TEMP_VAR, node)
                };
                let key_addr = self.declare_variable(key_var_name, key_node);

                self.push_instruction(Inst::load(iterable_addr), body);
                self.push_instruction(Inst::load(index_addr), body);
                self.push_instruction(Inst::LoadKey, key_node);
                self.push_instruction(Inst::store(key_addr), body);
                self.push_instruction(Inst::load(key_addr), key_node);

                let loop_exit_jump_ip = self.cur_inst_ptr();
                // Placeholder
                self.push_instruction(Inst::Nop, body);

                if let Some(item_node) = item {
                    let item_var_name = item_node.get_var_name().expect("Parsed correctly");
                    let item_addr = self.declare_variable(item_var_name, item_node);
                    self.push_instruction(Inst::load(iterable_addr), item_node);
                    self.push_instruction(Inst::load(key_addr), item_node);
                    self.push_instruction(Inst::LoadItem, item_node);
                    self.push_instruction(Inst::store(item_addr), item_node);
                }

                self.compile_block(body);

                let loop_continue_ip = self.cur_inst_ptr();
                self.push_instruction(Inst::load(index_addr), body);
                self.push_instruction(Inst::Incr, body);
                self.push_instruction(Inst::store(index_addr), body);

                (loop_next_iteration_ip, loop_continue_ip, loop_exit_jump_ip)
            } else if let AstNodeKind::WhileLoop(condition, body) = &node.kind {
                self.block_start(body, None, true);

                let loop_continue_ip = self.cur_inst_ptr();

                self.compile_expression(condition, None);

                let loop_exit_jump_ip = self.cur_inst_ptr();
                // Placeholder
                self.push_instruction(Inst::Nop, body);

                self.compile_block(body);

                (loop_continue_ip, loop_continue_ip, loop_exit_jump_ip)
            } else {
                panic!("Should be parsed correctly");
            };

        self.push_instruction(Inst::Jump(loop_next_iteration_ip), node);

        let loop_end_ip = self.cur_inst_ptr();
        *self.inst_mut(loop_exit_jump_ip) = if matches!(node.kind, AstNodeKind::ForLoop(..)) {
            Inst::JumpIfNull(loop_end_ip)
        } else {
            Inst::JumpIfFalsy(loop_end_ip)
        };

        let mut continues = Vec::new();
        std::mem::swap(&mut continues, &mut self.loop_data_mut().continues);
        let mut breaks = Vec::new();
        std::mem::swap(&mut breaks, &mut self.loop_data_mut().breaks);

        for continue_jump_ip in continues {
            // Increment before continuing
            *self.inst_mut(continue_jump_ip) = Inst::Jump(loop_continue_ip);
        }

        for break_jump_ip in breaks {
            *self.inst_mut(break_jump_ip) = Inst::Jump(loop_end_ip);
        }

        self.block_end();
    }

    fn compile_continue(&mut self, node: &'a AstNode) {
        let continue_ip = self.cur_inst_ptr();
        // Placeholder
        self.push_instruction(Inst::Nop, node);
        self.loop_data_mut().continues.push(continue_ip);
    }

    fn compile_break(&mut self, node: &'a AstNode) {
        let break_ip = self.cur_inst_ptr();
        // Placeholder
        self.push_instruction(Inst::Nop, node);
        self.loop_data_mut().breaks.push(break_ip);
    }

    fn compile_if_statement(&mut self, node: &'a AstNode) {
        let AstNodeKind::IfStatement(condition, block, else_block) = &node.kind else {
            unreachable!();
        };
        self.compile_expression(condition, None);

        let if_jump_ip = self.cur_inst_ptr();
        // Placeholder
        self.push_instruction(Inst::Nop, node);
        self.compile_block_full(block);

        if let Some(else_block) = else_block {
            let else_jump_ip = self.cur_inst_ptr();
            // Placeholder
            self.push_instruction(Inst::Nop, node);

            let else_start_ip = self.cur_inst_ptr();

            *self.inst_mut(if_jump_ip) = Inst::JumpIfFalsy(else_start_ip);

            self.compile_block_full(else_block);

            let after_else_ip = self.cur_inst_ptr();
            *self.inst_mut(else_jump_ip) = Inst::Jump(after_else_ip);
        } else {
            let after_if_ip = self.cur_inst_ptr();
            *self.inst_mut(if_jump_ip) = Inst::JumpIfFalsy(after_if_ip);
        }
    }

    fn block_start(&mut self, node: &'a AstNode, fn_node: Option<&'a AstNode>, is_loop: bool) {
        let AstNodeKind::Block(nodes) = &node.kind else {
            self.fatal("Expected block node", node);
        };

        let fn_data = fn_node.map(|_| FnData::default());
        let loop_data = is_loop.then(|| LoopData {
            breaks: Vec::new(),
            continues: Vec::new(),
        });

        self.scopes.push(ScopeData {
            declarations: Vec::new(),
            fn_data,
            loop_data,
        });
    }

    fn block_end(&mut self) {
        self.scopes
            .pop()
            .expect("Scope should exist when ending block");
    }

    fn loop_data_mut(&mut self) -> &mut LoopData {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(loop_data) = &mut scope.loop_data {
                return loop_data;
            }
        }
        panic!("No loop data found in current scopes");
    }

    fn fn_data(&self) -> Option<&FnData> {
        self.scopes
            .first()
            .expect("At least one scope exists")
            .fn_data
            .as_ref()
    }

    fn compile_block_full(&mut self, block: &'a AstNode) {
        self.block_start(block, None, false);
        self.compile_block(block);
        self.block_end();
    }

    fn compile_block(&mut self, block: &'a AstNode) {
        let AstNodeKind::Block(b) = &block.kind else {
            self.fatal("Expected block node", block);
        };
        for node in b.iter() {
            match &node.kind {
                AstNodeKind::DeclareAssign(..) | AstNodeKind::Assign(..) => {
                    self.compile_assignment(node)
                }
                AstNodeKind::FunctionDefinition(..) => self.compile_function_definition(node),
                AstNodeKind::FunctionCall(..) => self.compile_function_call(node, true),
                AstNodeKind::Return(..) => self.compile_return(node),
                AstNodeKind::ForLoop(..) | AstNodeKind::WhileLoop(..) => self.compile_loop(node),
                AstNodeKind::Continue => self.compile_continue(node),
                AstNodeKind::Break => self.compile_break(node),
                AstNodeKind::IfStatement(..) => self.compile_if_statement(node),
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
    c.compile_block_full(block);
    for (i, ins) in c.instructions.iter().enumerate() {
        trace!("{:4}: {}", i, ins);
    }
    c
}

pub fn binary_op_err(left_val: &Value, op: Operator, right_val: &Value) -> String {
    format!(
        "Cannot apply operator '{}' to operands {} and {})",
        op,
        left_val.dbg_display(),
        right_val.dbg_display()
    )
}

#[inline]
pub fn binary_op(l: &mut Value, op: Operator, r: &Value) -> OpResult {
    match op {
        Operator::Add => l.add(r),
        Operator::Sub => l.sub(r),
        Operator::Mul => l.mul(r),
        Operator::Div => l.div(r),
        Operator::Lt => l.lt(r),
        Operator::Gt => l.gt(r),
        Operator::Lte => l.lte(r),
        Operator::Gte => l.gte(r),
        Operator::Eq => l.eq_(r),
        Operator::Neq => l.neq(r),
        Operator::And => l.and(r),
        Operator::Or => l.or(r),
    }
}
