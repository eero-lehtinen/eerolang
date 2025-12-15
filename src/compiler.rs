use std::fmt::Display;

use foldhash::{HashMap, HashMapExt, HashSet, HashSetExt};
use log::trace;

use crate::{
    ast_parser::{AstNode, AstNodeKind, fatal_generic},
    builtins::{ArgsRequred, ProgramFn, all_builtins},
    tokenizer::{Literal, Operator, Token},
    value::{OpResult, Value},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstAddr(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalAddr(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalAddr(pub u32);

#[derive(Debug)]
pub enum Addr {
    Const(ConstAddr),
    Global(GlobalAddr),
    Local(LocalAddr),
}

impl Display for Addr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Addr::Const(addr) => write!(f, "{}", addr),
            Addr::Global(addr) => write!(f, "{}", addr),
            Addr::Local(addr) => write!(f, "{}", addr),
        }
    }
}
impl Display for ConstAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "C{}", self.0)
    }
}
impl Display for GlobalAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "G{}", self.0)
    }
}
impl Display for LocalAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "L{}", self.0)
    }
}

#[derive(Debug)]
pub enum Inst {
    #[allow(dead_code)]
    Nop,
    PushConst(ConstAddr),
    PushGlobal(GlobalAddr),
    PushLocal(LocalAddr),
    StoreGlobal(GlobalAddr),
    StoreLocal(LocalAddr),
    Pop,
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Lte,
    Gt,
    Gte,
    Eq,
    Neq,
    CallBuiltin(u32, u32), // function index, arg count
    Jump(u32),
    JumpIfFalsy(u32),
    JumpIfTruthy(u32),
}

impl Display for Inst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Inst::Nop => write!(f, "NOP"),
            Inst::PushConst(addr) => write!(f, "PUSH_CONST {}", addr.0),
            Inst::PushGlobal(addr) => write!(f, "PUSH_GLOBAL {}", addr.0),
            Inst::PushLocal(addr) => write!(f, "PUSH_LOCAL {}", addr.0),
            Inst::StoreGlobal(addr) => write!(f, "STORE_GLOBAL {}", addr.0),
            Inst::StoreLocal(addr) => write!(f, "STORE_LOCAL {}", addr.0),
            Inst::Pop => write!(f, "POP"),
            Inst::Add => write!(f, "ADD"),
            Inst::Sub => write!(f, "SUB"),
            Inst::Mul => write!(f, "MUL"),
            Inst::Div => write!(f, "DIV"),
            Inst::Lt => write!(f, "LT"),
            Inst::Lte => write!(f, "LTE"),
            Inst::Gt => write!(f, "GT"),
            Inst::Gte => write!(f, "GTE"),
            Inst::Eq => write!(f, "EQ"),
            Inst::Neq => write!(f, "NEQ"),
            Inst::CallBuiltin(func_index, arg_count) => {
                write!(f, "CALL_BUILTIN {} {}", func_index, arg_count)
            }
            Inst::Jump(target) => write!(f, "JUMP {}", target),
            Inst::JumpIfFalsy(target) => write!(f, "JUMP_IF_FALSY {}", target),
            Inst::JumpIfTruthy(target) => write!(f, "JUMP_IF_TRUTHY {}", target),
        }
    }
}

impl Inst {
    fn binary_op(op: Operator) -> Self {
        match op {
            Operator::Add => Inst::Add,
            Operator::Sub => Inst::Sub,
            Operator::Mul => Inst::Mul,
            Operator::Div => Inst::Div,
            Operator::Lt => Inst::Lt,
            Operator::Lte => Inst::Lte,
            Operator::Gt => Inst::Gt,
            Operator::Gte => Inst::Gte,
            Operator::Eq => Inst::Eq,
            Operator::Neq => Inst::Neq,
            Operator::And => panic!("And is not an instruction"),
            Operator::Or => panic!("Or is not an instruction"),
        }
    }
}

#[derive(Debug)]
struct ScopeData<'a> {
    declarations: Vec<&'a str>,
    /// Stored at the root of the scope stack for function scopes.
    fn_data: Option<FnData>,
    // loop_data: Option<LoopData>,
}

#[derive(Debug, Default)]
struct FnData {
    // frame_ptr: u32,
    /// Keeps track of variables declared in the function to figure out stack allocation for locals.
    /// Holds (name, scope depth)
    locals: Vec<(String, u32)>,
}

#[derive(Debug)]
struct LoopData {
    frame_ptr: u32,
    stack_ptr: u32,
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

impl<'a> Compilation<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        let mut builtins = HashMap::new();
        for (i, (name, func, args)) in all_builtins().iter().enumerate() {
            builtins.insert(name.to_string(), (*func, i, *args));
        }
        Compilation {
            instructions: Vec::new(),
            constants: Vec::new(),
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
                self.push_instruction(Inst::PushConst(addr), expr);
            }
            AstNodeKind::Variable(name) => {
                let addr = self.variable_addr(name, expr, to_decl);
                match addr {
                    Addr::Const(_) => self.fatal("Cannot use constant as variable", expr),
                    Addr::Global(addr) => {
                        self.push_instruction(Inst::PushGlobal(addr), expr);
                    }
                    Addr::Local(addr) => {
                        self.push_instruction(Inst::PushLocal(addr), expr);
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
        todo!();
        // let AstNodeKind::FunctionDefinition(name, args, body) = &node.kind else {
        //     unreachable!();
        // };
        //
        // if all_builtins().iter().any(|(n, _, _)| n == name) {
        //     self.fatal(
        //         &format!("Cannot redefine built-in function '{}'", name),
        //         node,
        //     );
        // }
        // if self.functions.contains_key(name) {
        //     self.fatal(&format!("Function '{}' is already defined", name), node);
        // }
        //
        // let fn_skip_jump_ip = self.cur_inst_ptr();
        // // Function instructions are defined "in the middle" of the instructions so we need to skip
        // // over it to make top level code work correctly.
        // self.push_instruction(Inst::jump(0), node);
        //
        // let fn_start_ip = self.cur_inst_ptr();
        //
        // let args_required = ArgsRequred::Exact(args.len() as u32);
        // self.functions
        //     .insert(name.clone(), (fn_start_ip, args_required));
        //
        // self.block_start(body, Some(node), None, false);
        //
        // // Load arguments from argument registers to stack variables
        // for (arg_idx, arg) in args.iter().enumerate() {
        //     let arg_name = arg.get_var_name().expect("Parsed correctly");
        //     let arg_addr = self.variable_addr(arg_name, arg, None);
        //     let arg_reg = Addr::abs(ARG_REG_START + arg_idx as u32);
        //     self.push_instruction(Inst::load_addr(arg_addr, arg_reg), node);
        // }
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
        //
        // // Store temporaries to survive the function call.
        // let save_args_count = args.len() as u32;
        // let save_regs_count = REGS_TO_STORE_ON_FN_CALL.len() as u32;
        // self.push_instruction(Inst::save_regs(save_args_count), node);
        // self.cur_stack_ptr_offset += save_regs_count + save_args_count;
        //
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

    fn compile_loop(&mut self, node: &'a AstNode) {
        todo!()
        // let (body, loop_continue_ip, loop_exit_inst_index, index_addr) =
        //     if let AstNodeKind::ForLoop(key, item, collection, body) = &node.kind {
        //         self.block_start(body, None, Some(node), false);
        //
        //         let iterable_addr = self.variable_addr(Self::FOR_ITERABLE_TEMP_VAR, node, None);
        //
        //         self.compile_expression(collection, Some(Self::FOR_ITERABLE_TEMP_VAR));
        //         self.push_instruction(Inst::pop(iterable_addr), collection);
        //
        //         self.push_instruction(Inst::init_map_iteration_list(iterable_addr), node);
        //
        //         let index_addr = self.variable_addr(Self::FOR_INDEX_TEMP_VAR, node, None);
        //         self.push_instruction(Inst::load_int(index_addr, 0), node);
        //
        //         let loop_continue_ip = self.cur_inst_ptr();
        //
        //         let (key_var_name, key_node) = if let Some(key_node) = key {
        //             (
        //                 key_node.get_var_name().expect("Parsed correctly"),
        //                 key_node.as_ref(),
        //             )
        //         } else {
        //             (Self::FOR_KEY_TEMP_VAR, node)
        //         };
        //         let key_addr = self.variable_addr(key_var_name, key_node, None);
        //         self.push_instruction(
        //             Inst::load_iteration_key(key_addr, iterable_addr, index_addr),
        //             key_node,
        //         );
        //
        //         let loop_exit_inst_index = self.cur_inst_ptr();
        //         // Placeholder
        //         self.push_instruction(Inst::jump_if_zero(0, SUCCESS_FLAG_REG), node);
        //
        //         if let Some(item_node) = item {
        //             let item_var_name = item_node.get_var_name().expect("Parsed correctly");
        //             let item_addr = self.variable_addr(item_var_name, item_node, None);
        //             self.push_instruction(
        //                 Inst::load_collection_item(item_addr, iterable_addr, key_addr),
        //                 item_node,
        //             );
        //         }
        //
        //         self.compile_block(body);
        //
        //         self.push_instruction(Inst::incr(index_addr), node);
        //
        //         (
        //             body,
        //             Some(loop_continue_ip),
        //             Some(loop_exit_inst_index),
        //             Some(index_addr),
        //         )
        //     } else if let AstNodeKind::WhileLoop(condition, body) = &node.kind {
        //         self.block_start(body, None, None, true);
        //
        //         let loop_continue_ip = self.cur_inst_ptr();
        //
        //         let (cond_addr, cond_val) = self.compile_expression(condition, None);
        //
        //         let const_cond_true = cond_val.map(|v| !v.is_falsy());
        //
        //         let loop_exit_inst_index = self.cur_inst_ptr();
        //         // Placeholder
        //         self.push_instruction(Inst::jump_if_zero(0, cond_addr), node);
        //
        //         self.compile_block(body);
        //
        //         (
        //             body,
        //             Some(loop_continue_ip),
        //             Some(loop_exit_inst_index),
        //             None,
        //         )
        //     } else {
        //         panic!("Should be parsed correctly");
        //     };
        // if let Some(loop_continue_ip) = loop_continue_ip {
        //     self.push_instruction(Inst::jump(loop_continue_ip), node);
        // }
        //
        // if let Some(loop_exit_inst_index) = loop_exit_inst_index {
        //     let loop_end_ip = self.cur_inst_ptr();
        //     self.inst_mut(loop_exit_inst_index)
        //         .set_jump_target(loop_end_ip);
        // }
        //
        // let mut continues = Vec::new();
        // std::mem::swap(&mut continues, &mut self.loop_data_mut().continues);
        // let mut breaks = Vec::new();
        // std::mem::swap(&mut breaks, &mut self.loop_data_mut().breaks);
        //
        // if let Some(loop_continue_ip) = loop_continue_ip {
        //     for continue_index in continues {
        //         if let Some(index_addr) = index_addr {
        //             // Increment before continuing
        //             self.inst_mut(continue_index).set_incr_dst(index_addr);
        //             self.inst_mut(continue_index + 1)
        //                 .set_jump_target(loop_continue_ip);
        //         } else {
        //             self.inst_mut(continue_index)
        //                 .set_jump_target(loop_continue_ip);
        //         }
        //     }
        // }
        //
        // self.block_end(body);
        //
        // let loop_end_after_sp_reset_ip = self.cur_inst_ptr();
        //
        // for break_index in breaks {
        //     self.inst_mut(break_index)
        //         .set_jump_target(loop_end_after_sp_reset_ip);
        // }
    }

    fn compile_continue(&mut self, node: &'a AstNode) {
        todo!();
        // let sub_sp = self.cur_stack_ptr_offset - self.loop_data().stack_ptr;
        // if sub_sp > 0 {
        //     self.push_instruction(Inst::sub_stack_pointer(sub_sp), node);
        // }
        // let continue_ip = self.cur_inst_ptr();
        // self.loop_data_mut().continues.push(continue_ip);
        // // Placeholder
        // self.push_instruction(Inst::incr(Addr::abs(0)), node);
        // // Placeholder
        // self.push_instruction(Inst::jump(0), node);
    }

    fn compile_break(&mut self, node: &'a AstNode) {
        todo!();
        // let sub_sp = self.cur_stack_ptr_offset - self.loop_data().frame_ptr;
        // if sub_sp > 0 {
        //     self.push_instruction(Inst::sub_stack_pointer(sub_sp), node);
        // }
        // let break_ip = self.cur_inst_ptr();
        // // Placeholder
        // self.push_instruction(Inst::jump(0), node);
        // self.loop_data_mut().breaks.push(break_ip);
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

    // // If the iterable is an expression, it needs to be stored somwhere.
    // const FOR_ITERABLE_TEMP_VAR: &'static str = "__for_iterable_temp";
    // // Index needs to be stored somewhere.
    // const FOR_INDEX_TEMP_VAR: &'static str = "__for_index_temp";
    // // Even if not assigned to a variable, the key needs to be stored somewhere.
    // const FOR_KEY_TEMP_VAR: &'static str = "__for_key_temp";

    fn block_start(
        &mut self,
        node: &'a AstNode,
        fn_node: Option<&'a AstNode>,
        for_loop_node: Option<&'a AstNode>,
        while_loop: bool,
    ) {
        let AstNodeKind::Block(nodes) = &node.kind else {
            self.fatal("Expected block node", node);
        };

        // let mut cur_scope_var_decls: Vec<(&'a str, usize)> = Vec::new();
        // macro_rules! add_decl_node {
        //     ($decl:expr) => {
        //         let var_name = $decl.get_var_name().expect("Parsed correctly");
        //         if cur_scope_var_decls.iter().any(|(v, _)| *v == var_name) {
        //             self.fatal(
        //                 &format!("Variable '{}' already declared in this scope", var_name),
        //                 $decl,
        //             );
        //         }
        //         let token = &self.tokens[$decl.token_idx];
        //         cur_scope_var_decls.push((var_name, token.byte_pos_start));
        //     };
        // }
        //
        // if let Some(node) = fn_node {
        //     let AstNodeKind::FunctionDefinition(_, args, _) = &node.kind else {
        //         panic!("Should be parsed correctly");
        //     };
        //
        //     for arg in args {
        //         add_decl_node!(arg);
        //     }
        // }
        //
        // // Add loop variable declarations
        // if let Some(loop_node) = for_loop_node {
        //     let AstNodeKind::ForLoop(key, item, _, _) = &loop_node.kind else {
        //         panic!("Should be parsed correctly");
        //     };
        //
        //     let token = &self.tokens[loop_node.token_idx];
        //     // These are always not named by the user.
        //     cur_scope_var_decls.extend_from_slice(&[
        //         (Self::FOR_ITERABLE_TEMP_VAR, token.byte_pos_start),
        //         (Self::FOR_INDEX_TEMP_VAR, token.byte_pos_start),
        //     ]);
        //
        //     // This is needed but allowed to be underscore by the user.
        //     if let Some(key_node) = key {
        //         add_decl_node!(key_node);
        //     } else {
        //         cur_scope_var_decls.push((Self::FOR_KEY_TEMP_VAR, token.byte_pos_start));
        //     }
        //
        //     // This doesn't even need to be created if it's not set or underscore.
        //     if let Some(item_node) = item {
        //         add_decl_node!(item_node);
        //     }
        // }
        //
        // for node in nodes {
        //     if matches!(&node.kind, AstNodeKind::DeclareAssign(_, _)) {
        //         add_decl_node!(node);
        //     }
        // }

        // let frame_ptr = self.cur_stack_ptr_offset;

        // let add_sp = cur_scope_var_decls.len() as u32;
        // if add_sp > 0 {
        //     self.push_instruction(Inst::add_stack_pointer(add_sp), node);
        //     self.cur_stack_ptr_offset += add_sp;
        // }

        let fn_data = fn_node.map(|_| FnData::default());
        //
        // let loop_data = if for_loop_node.is_some() || while_loop {
        //     Some(LoopData {
        //         frame_ptr,
        //         stack_ptr: self.cur_stack_ptr_offset,
        //         breaks: Vec::new(),
        //         continues: Vec::new(),
        //     })
        // } else {
        //     None
        // };

        self.scopes.push(ScopeData {
            declarations: Vec::new(),
            fn_data,
            // frame_ptr,
            // fn_data,
            // loop_data,
        });
    }

    fn block_end(&mut self, node: &'a AstNode) {
        self.scopes
            .pop()
            .expect("Scope should exist when ending block");
        // let sub_sp = self.cur_stack_ptr_offset - scope.frame_ptr;
        // if sub_sp > 0 {
        //     self.push_instruction(Inst::sub_stack_pointer(sub_sp), node);
        //     self.cur_stack_ptr_offset -= sub_sp;
        // }
    }

    fn loop_data(&self) -> &LoopData {
        // for scope in self.scopes.iter().rev() {
        //     if let Some(loop_data) = &scope.loop_data {
        //         return loop_data;
        //     }
        // }
        panic!("No loop data found in current scopes");
    }

    fn loop_data_mut(&mut self) -> &mut LoopData {
        // for scope in self.scopes.iter_mut().rev() {
        //     if let Some(loop_data) = &mut scope.loop_data {
        //         return loop_data;
        //     }
        // }
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
        self.block_start(block, None, None, false);
        self.compile_block(block);
        self.block_end(block);
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
