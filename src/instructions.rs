use std::fmt::Display;

use crate::tokenizer::Operator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstAddr(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalAddr(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalAddr(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    Nop,
    /// Takes value from constant slot and puts it on eval stack.
    LoadConst(ConstAddr),
    /// Takes value from global slot and puts it on eval stack.
    LoadGlobal(GlobalAddr),
    /// Takes value from local slot and puts it on eval stack.
    LoadLocal(LocalAddr),
    /// Takes value from eval stack and puts it into global slot.
    StoreGlobal(GlobalAddr),
    /// Takes value from eval stack and puts it into local slot.
    StoreLocal(LocalAddr),
    Pop,
    InitMapIter,
    LoadKey,
    LoadItem,
    Incr,
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
    JumpIfNull(u32),
    JumpIfFalsy(u32),
    JumpIfTruthy(u32),
}

impl Inst {
    pub fn store(addr: Addr) -> Self {
        match addr {
            Addr::Const(_) => panic!("Cannot store to constant address"),
            Addr::Global(gaddr) => Self::StoreGlobal(gaddr),
            Addr::Local(laddr) => Self::StoreLocal(laddr),
        }
    }

    pub fn load(addr: Addr) -> Self {
        match addr {
            Addr::Const(caddr) => Self::LoadConst(caddr),
            Addr::Global(gaddr) => Self::LoadGlobal(gaddr),
            Addr::Local(laddr) => Self::LoadLocal(laddr),
        }
    }

    pub fn binary_op(op: Operator) -> Self {
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

impl Display for Inst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Inst::Nop => write!(f, "NOP"),
            Inst::LoadConst(addr) => write!(f, "PUSH_CONST {}", addr.0),
            Inst::LoadGlobal(addr) => write!(f, "PUSH_GLOBAL {}", addr.0),
            Inst::LoadLocal(addr) => write!(f, "PUSH_LOCAL {}", addr.0),
            Inst::StoreGlobal(addr) => write!(f, "STORE_GLOBAL {}", addr.0),
            Inst::StoreLocal(addr) => write!(f, "STORE_LOCAL {}", addr.0),
            Inst::Pop => write!(f, "POP"),
            Inst::InitMapIter => write!(f, "INIT_MAP_ITER"),
            Inst::LoadKey => write!(f, "LOAD_KEY"),
            Inst::LoadItem => write!(f, "LOAD_ITEM"),
            Inst::Incr => write!(f, "INCR"),
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
            Inst::JumpIfNull(target) => write!(f, "JUMP_IF_NULL {}", target),
            Inst::JumpIfFalsy(target) => write!(f, "JUMP_IF_FALSY {}", target),
            Inst::JumpIfTruthy(target) => write!(f, "JUMP_IF_TRUTHY {}", target),
        }
    }
}
