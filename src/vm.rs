use std::rc::Rc;

use foldhash::{HashMap, HashMapExt};
use log::trace;

use crate::{
    ast_parser::{AstNode, AstNodeKind},
    builtins::{ProgramFn, builtin_get, get_builtins},
    tokenizer::{Operator, Value},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Addr {
    Abs(u32),
    Stack(u32),
}

#[derive(Debug, Clone)]
enum Inst {
    Load {
        dst: Addr,
        src: Addr,
    },
    LoadFromCollection {
        dst: Addr,
        collection: Addr,
        index: Addr,
    },
    Push {
        src: Addr,
    },
    Pop {
        dst: Addr,
    },
    BinaryOp {
        op: Operator,
        dst: Addr,
        src1: Addr,
        src2: Addr,
    },
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

const RESULT_REG1: Addr = Addr::Abs(0);
const RESULT_REG2: Addr = Addr::Abs(1);
const ZERO_REG: Addr = Addr::Abs(2);
const TRASH_REG: Addr = Addr::Abs(3);
const ARG_REG_START: u32 = 10;
const ARG_REG_COUNT: u32 = 10;
const RESERVED_REGS: u32 = 20;
const STACK_SIZE: u32 = 2 << 11;

pub struct Compilation {
    instructions: Vec<Inst>,
    variables: HashMap<Rc<str>, Addr>,
    literals: Vec<(Value, Addr)>,
    functions: HashMap<Rc<str>, (ProgramFn, usize)>,
}

impl Compilation {
    fn new() -> Self {
        let mut functions = HashMap::new();
        for (i, (name, func)) in get_builtins().iter().enumerate() {
            functions.insert(Rc::from(*name), (*func, i));
        }
        Compilation {
            instructions: Vec::new(),
            variables: HashMap::new(),
            literals: Vec::new(),
            functions,
        }
    }

    fn cur_inst_ptr(&self) -> usize {
        self.instructions.len()
    }

    fn variable_addr(&mut self, name: &Rc<str>) -> Addr {
        if let Some(&addr) = self.variables.get(name) {
            addr
        } else {
            let addr = Addr::Abs(self.variables.len() as u32 + RESERVED_REGS + STACK_SIZE);
            self.variables.insert(Rc::clone(name), addr);
            addr
        }
    }

    fn make_literal(&mut self, value: &Value) -> Addr {
        let lit_name = Rc::from(format!("__lit_{}", self.literals.len()));
        let addr = self.variable_addr(&lit_name);
        self.literals.push((value.clone(), addr));
        addr
    }

    fn compile_function_args(&mut self, args: &[AstNode]) {
        if args.len() > ARG_REG_COUNT as usize {
            panic!("Too many arguments in function call");
        }
        for (i, arg) in args.iter().enumerate() {
            let arg_reg = Addr::Abs(ARG_REG_START + i as u32);
            let res_reg = self.compile_expression(arg, arg_reg);
            if res_reg != arg_reg {
                self.instructions.push(Inst::Load {
                    dst: arg_reg,
                    src: res_reg,
                });
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
            self.instructions.push(Inst::Load {
                dst: Addr::Abs(target_reg),
                src: Addr::Abs(arg_reg),
            });
        }
    }

    fn compile_function_call(&mut self, name: &Rc<str>, arg_count: usize, dst: Addr) -> Addr {
        let (_, func_index) = self
            .functions
            .get(name)
            .unwrap_or_else(|| panic!("Undefined function: {}", name));

        self.instructions.push(Inst::Call {
            dst,
            func: *func_index,
            arg_count: arg_count as u8,
        });

        dst
    }

    fn compile_expression(&mut self, expr: &AstNode, dst_suggestion: Addr) -> Addr {
        match &expr.kind {
            AstNodeKind::Literal(value) => self.make_literal(value),
            AstNodeKind::Variable(name) => self.variable_addr(name),
            AstNodeKind::BinaryOp(left, op, right) => {
                let left_reg = self.compile_expression(left, RESULT_REG1);
                let right_reg = self.compile_expression(right, RESULT_REG2);
                self.instructions.push(Inst::BinaryOp {
                    op: *op,
                    dst: dst_suggestion,
                    src1: left_reg,
                    src2: right_reg,
                });
                dst_suggestion
            }
            AstNodeKind::FunctionCall(name, args) => {
                self.compile_function_args(args);
                self.compile_function_call(name, args.len(), dst_suggestion)
            }
            _ => todo!(),
        }
    }

    fn compile_block(&mut self, block: &[AstNode]) {
        for node in block.iter() {
            match &node.kind {
                AstNodeKind::Assign(var, expr) => {
                    let var_reg = self.variable_addr(var);
                    let expr_reg = self.compile_expression(expr, var_reg);
                    if expr_reg != var_reg {
                        self.instructions.push(Inst::Load {
                            dst: var_reg,
                            src: expr_reg,
                        });
                    }
                }
                AstNodeKind::FunctionCall(name, args) => {
                    self.compile_function_args(args);
                    self.compile_function_call(name, args.len(), RESULT_REG1);
                }
                AstNodeKind::ForLoop(index_var, item_var, collection, body) => {
                    let iterable_addr = self.compile_expression(collection, RESULT_REG1);

                    self.instructions.push(Inst::Push { src: iterable_addr });
                    self.instructions.push(Inst::Push { src: ZERO_REG });
                    let iterable_addr = Addr::Stack(1);
                    let index_addr = Addr::Stack(0);

                    let for_cmp_ip = self.cur_inst_ptr();
                    self.instructions.push(Inst::JumpIfNotInRange {
                        target: 0, // Placeholder, will be filled later
                        range: iterable_addr,
                        src: index_addr,
                    });

                    if index_var.as_ref() != "_" {
                        let index_reg = self.variable_addr(index_var);
                        self.instructions.push(Inst::Load {
                            dst: index_reg,
                            src: index_addr,
                        });
                    }

                    if item_var.as_ref().is_some_and(|v| v.as_ref() != "_") {
                        let item_reg = self.variable_addr(item_var.as_ref().unwrap());
                        self.instructions.push(Inst::LoadFromCollection {
                            dst: item_reg,
                            collection: iterable_addr,
                            index: index_addr,
                        });
                    }

                    self.compile_block(body);

                    self.instructions.push(Inst::Incr { dst: index_addr });
                    self.instructions.push(Inst::Jump { target: for_cmp_ip });

                    let loop_end_ip = self.cur_inst_ptr();
                    self.instructions[for_cmp_ip].set_jump_target(loop_end_ip);

                    self.instructions.push(Inst::Pop { dst: TRASH_REG });
                    self.instructions.push(Inst::Pop { dst: TRASH_REG });
                }
                AstNodeKind::IfStatement(condition, block, else_block) => {
                    let cond_reg = self.compile_expression(condition, RESULT_REG1);

                    let if_jump_ip = self.cur_inst_ptr();
                    self.instructions.push(Inst::JumpIfZero {
                        target: 0, // Placeholder
                        cond: cond_reg,
                    });

                    self.compile_block(block);

                    if !else_block.is_empty() {
                        let else_jump_ip = self.cur_inst_ptr();
                        self.instructions.push(Inst::Jump {
                            target: 0, // Placeholder
                        });

                        let else_start_ip = self.cur_inst_ptr();
                        self.instructions[if_jump_ip].set_jump_target(else_start_ip);

                        self.compile_block(else_block);

                        let after_else_ip = self.cur_inst_ptr();
                        self.instructions[else_jump_ip].set_jump_target(after_else_ip);
                    } else {
                        let after_if_ip = self.cur_inst_ptr();
                        self.instructions[if_jump_ip].set_jump_target(after_if_ip);
                    }
                }
                _ => {
                    panic!("Unsupported AST node in compile_block: {:?}", node.kind);
                }
            }
        }
    }
}

#[allow(dead_code)]
pub fn compile(block: &[AstNode]) -> Compilation {
    let mut c = Compilation::new();
    c.compile_block(block);
    for ins in c.instructions.iter() {
        trace!("{:?}", ins);
    }
    c
}

#[allow(dead_code)]
pub struct Vm {
    instructions: Vec<Inst>,
    ip: usize,
    sp: usize,
    memory: Vec<Value>,
    functions: Vec<ProgramFn>,
}

fn placeholder_func(_: &[Value]) -> Result<Value, String> {
    Err("Placeholder function called".to_string())
}

#[allow(dead_code)]
impl Vm {
    pub fn new(ctx: Compilation) -> Self {
        let mut memory = vec![
            Value::Integer(0);
            ctx.variables.len() + RESERVED_REGS as usize + STACK_SIZE as usize
        ];
        for (lit_value, lit_reg) in ctx.literals.iter() {
            let Addr::Abs(lit_reg) = *lit_reg else {
                panic!("Literal register is not absolute");
            };
            memory[lit_reg as usize] = lit_value.clone();
        }

        let mut functions = vec![placeholder_func as ProgramFn; ctx.functions.len()];
        for (_, (func, index)) in ctx.functions.iter() {
            functions[*index] = *func;
        }

        Vm {
            instructions: ctx.instructions,
            ip: 0,
            memory,
            functions,
            sp: RESERVED_REGS as usize,
        }
    }

    fn mem(&self, addr: Addr) -> usize {
        match addr {
            Addr::Abs(reg) => reg as usize,
            Addr::Stack(offset) => {
                debug_assert!(self.sp >= (offset + RESERVED_REGS) as usize);
                self.sp - offset as usize
            }
        }
    }

    fn mem_get(&self, addr: Addr) -> &Value {
        let pos = self.mem(addr);
        debug_assert!(pos < self.memory.len());
        unsafe { self.memory.get_unchecked(pos) }
    }

    fn mem_set(&mut self, addr: Addr, value: Value) {
        let pos = self.mem(addr);
        debug_assert!(pos < self.memory.len());
        unsafe {
            *self.memory.get_unchecked_mut(pos) = value;
        }
    }

    pub fn run(&mut self) {
        while self.ip < self.instructions.len() {
            // trace!("IP {}: {:?}", self.ip, self.instructions[self.ip]);
            // trace!(
            //     "SP {}\n {:?}",
            //     self.sp,
            //     self.memory[RESERVED_REGS..self.sp + 1].to_vec()
            // );
            match self.instructions[self.ip] {
                Inst::Load { dst, src } => {
                    self.mem_set(dst, self.mem_get(src).clone());
                }
                Inst::LoadFromCollection {
                    dst,
                    collection,
                    index,
                } => {
                    // TODO: this might be not good
                    let collection = self.mem_get(collection);
                    let index = self.mem_get(index);
                    self.mem_set(
                        dst,
                        builtin_get(&[collection.clone(), index.clone()])
                            .expect("builtin_get failed"),
                    );
                }
                Inst::Push { src } => {
                    self.sp += 1;
                    let val = self.mem_get(src).clone();
                    self.mem_set(Addr::Stack(0), val);
                    debug_assert!(self.sp < RESERVED_REGS as usize + STACK_SIZE as usize);
                }
                Inst::Pop { dst } => {
                    let val = self.mem_get(Addr::Stack(0)).clone();
                    self.mem_set(dst, val);
                    self.sp -= 1;
                    debug_assert!(self.sp >= RESERVED_REGS as usize);
                }
                Inst::BinaryOp {
                    op,
                    dst,
                    src1,
                    src2,
                } => {
                    let res = Vm::binary_op(self.mem_get(src1), op, self.mem_get(src2));
                    self.mem_set(dst, res);
                }
                Inst::Call {
                    dst,
                    func,
                    arg_count,
                } => {
                    self.call_function(dst, func, arg_count);
                }
                Inst::Incr { dst } => {
                    if let Value::Integer(i) = self.mem_get(dst) {
                        self.mem_set(dst, Value::Integer(i + 1));
                    } else {
                        panic!("Cannot increment non-integer value in addr {:?}", dst);
                    }
                }
                Inst::JumpIfNotInRange { target, range, src } => {
                    let iterable_value = self.mem_get(range);
                    let index = self.mem_get(src);

                    let index = match index {
                        Value::Integer(i) => *i,
                        _ => panic!("Cannot compare non-integer value in addr {:?}", src),
                    };

                    let in_range = match iterable_value {
                        Value::List(list_rc) => {
                            let list = list_rc.borrow();
                            index >= 0 && index < list.len() as i64
                        }
                        Value::Range(r) => index >= r.start && index < r.end,
                        _ => panic!("Cannot get length of non-list value in addr {:?}", range),
                    };
                    if !in_range {
                        self.ip = target;
                        continue;
                    }
                }
                Inst::Jump { target } => {
                    self.ip = target;
                    continue;
                }
                Inst::JumpIfZero { target, cond } => {
                    let cond_value = &self.mem_get(cond);
                    let is_zero = match cond_value {
                        Value::Integer(i) => *i == 0,
                        _ => panic!("Cannot evaluate truthiness of value in register {:?}", cond),
                    };
                    if is_zero {
                        self.ip = target;
                        continue;
                    }
                }
            }
            self.ip += 1;
        }
    }

    fn call_function(&mut self, dst: Addr, func: usize, arg_count: u8) {
        let func_impl = &self.functions[func];
        let args =
            &mut self.memory[ARG_REG_START as usize..ARG_REG_START as usize + arg_count as usize];
        let result = func_impl(args).expect("Function call failed");
        self.mem_set(dst, result);
    }

    fn binary_op(left_val: &Value, op: Operator, right_val: &Value) -> Value {
        macro_rules! unsupported {
            () => {
                panic!(
                    "Cannot apply operator {:?} to operands (left: {:?}, right: {:?})",
                    op, left_val, right_val,
                )
                // self.fatal(
                //     &format!(
                //         "Cannot apply operator {:?} to operands (left: {:?}, right: {:?})",
                //         op, left_val, right_val,
                //     ),
                //     expr,
                // )
            };
        }

        if let (Value::String(l), Value::String(r)) = (&left_val, &right_val) {
            return match op {
                Operator::Plus => Value::String(Rc::from(l.as_ref().to_owned() + r.as_ref())),
                Operator::Eq => Value::Integer((l == r) as i64),
                Operator::Neq => Value::Integer((l != r) as i64),
                _ => unsupported!(),
            };
        }

        if let (Value::Integer(l), Value::Integer(r)) = (&left_val, &right_val) {
            return Value::Integer(match op {
                Operator::Plus => l + r,
                Operator::Minus => l - r,
                Operator::Multiply => l * r,
                Operator::Divide => l / r,
                Operator::Lt => (l < r) as i64,
                Operator::Gt => (l > r) as i64,
                Operator::Lte => (l <= r) as i64,
                Operator::Gte => (l >= r) as i64,
                Operator::Eq => (l == r) as i64,
                Operator::Neq => (l != r) as i64,
            });
        }

        // Promote to float if both weren't integers
        let right_prom = if let Value::Integer(i) = &right_val {
            Value::Float(*i as f64)
        } else {
            right_val.clone()
        };
        let left_prom = if let Value::Integer(i) = &left_val {
            Value::Float(*i as f64)
        } else {
            left_val.clone()
        };

        if let (Value::Float(l), Value::Float(r)) = (&left_prom, &right_prom) {
            return match op {
                Operator::Plus => Value::Float(l + r),
                Operator::Minus => Value::Float(l - r),
                Operator::Multiply => Value::Float(l * r),
                Operator::Divide => Value::Float(l / r),
                Operator::Lt => Value::Integer((l < r) as i64),
                Operator::Gt => Value::Integer((l > r) as i64),
                Operator::Lte => Value::Integer((l <= r) as i64),
                Operator::Gte => Value::Integer((l >= r) as i64),
                Operator::Eq => Value::Integer((l == r) as i64),
                Operator::Neq => Value::Integer((l != r) as i64),
            };
        }

        unsupported!();
    }
}
