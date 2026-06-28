#![allow(clippy::mutable_key_type)]
//! Embedding API for the eerolang interpreter.
//!
//! The typical flow is:
//!
//! ```
//! use eerolang::{compile, Value};
//!
//! let mut program = compile("x := 1 + 2", &[]).unwrap();
//! program.run().unwrap();
//! assert_eq!(program.global("x").unwrap().as_number(), Some(3.0));
//! ```

use std::mem::ManuallyDrop;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Once, OnceLock};

use bumpalo::Bump;
use ouroboros::self_referencing;

pub mod ast_parser;
pub mod builtins;
pub mod compiler;
pub mod instructions;
pub mod tokenizer;
pub mod value;
pub mod vm;

pub use value::{Value, ValueRef};

use tokenizer::Token;
use vm::Vm;

/// Arguments exposed to programs via the `args()` builtin. Unset means empty.
pub static EXTRA_ARGS: OnceLock<Vec<String>> = OnceLock::new();

/// Panic payload for fatal language errors, caught and surfaced as `Err`.
pub struct LangError(pub String);

/// Suppresses the default panic message for [`LangError`] (its diagnostic is
/// already on stderr). Idempotent.
pub fn install_panic_hook() {
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let default = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            if info.payload().is::<LangError>() {
                return;
            }
            default(info);
        }));
    });
}

/// Runs `f`, turning a [`LangError`] panic into `Err`; other panics propagate.
fn catch_lang<R>(f: impl FnOnce() -> R) -> Result<R, String> {
    install_panic_hook();
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(value) => Ok(value),
        Err(payload) => match payload.downcast::<LangError>() {
            Ok(err) => Err(err.0),
            Err(other) => panic::resume_unwind(other),
        },
    }
}

/// Self-referential core: owns the arena, source, and token buffer, with the
/// `Vm` borrowing from all three.
#[self_referencing]
struct ProgramInner {
    bump: Bump,
    source: String,
    #[borrows(bump, source)]
    #[covariant]
    tokens: Vec<Token<'this>>,
    #[borrows(bump, source, tokens)]
    #[covariant]
    vm: Vm<'this>,
}

pub struct Program {
    // `ManuallyDrop` so `Drop` can tear down the VM (releasing its GC refs)
    // before forcing a collection.
    inner: ManuallyDrop<ProgramInner>,
}

/// Compiles `script` into a reusable [`Program`].
///
/// `global_names` are pre-declared as input globals the script may read; supply
/// their values via [`run_with_globals`](Program::run_with_globals), or
/// [`run`](Program::run) leaves them null. Returns `Err` on a compile failure.
pub fn compile(script: &str, global_names: &[&str]) -> Result<Program, String> {
    catch_lang(|| build_program(script, global_names))
}

fn build_program(script: &str, global_names: &[&str]) -> Program {
    let inner = ProgramInnerBuilder {
        bump: Bump::new(),
        source: script.to_string(),
        tokens_builder: |bump, source| tokenizer::tokenize(bump, source, false),
        vm_builder: |bump, source, tokens| {
            let predeclared: Vec<&str> = global_names
                .iter()
                .map(|name| &*bump.alloc_str(name))
                .collect();
            let block = ast_parser::parse(bump, source, tokens);
            let compilation = compiler::compile_with_globals(block, source, tokens, &predeclared);
            Vm::new(compilation)
        },
    }
    .build();
    Program {
        inner: ManuallyDrop::new(inner),
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        // SAFETY: `inner` is never touched again. Drop it first (tearing down
        // the VM and releasing its GC references), then collect.
        unsafe { ManuallyDrop::drop(&mut self.inner) };
        // Reclaim this program's now-unreachable GC values.
        dumpster::unsync::collect();
    }
}

impl Program {
    /// Runs from a clean state, all globals (including inputs) start null.
    pub fn run(&mut self) -> Result<(), String> {
        self.inner.with_vm_mut(|vm| {
            vm.reset();
            catch_lang(AssertUnwindSafe(|| vm.run(false)))
        })
    }

    /// Runs from a clean state after setting the given input globals. Only names
    /// from [`compile`] are allowed, others return `Err` without running.
    pub fn run_with_globals(&mut self, globals: &[(&str, Value)]) -> Result<(), String> {
        self.inner.with_vm_mut(|vm| {
            vm.reset();
            for (name, value) in globals {
                if !vm.set_input_global(name, value.clone()) {
                    return Err(format!("'{}' is not an input global of this program", name));
                }
            }
            catch_lang(AssertUnwindSafe(|| vm.run(false)))
        })
    }

    /// A clone of the named global's value, or `None` if absent.
    pub fn global(&self, name: &str) -> Option<Value> {
        self.inner.borrow_vm().get_global(name).cloned()
    }

    /// Every global as `(name, value)` in declaration order.
    pub fn globals(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.inner.borrow_vm().global_entries()
    }

    /// The input global names supplied to [`compile`], in declaration order.
    pub fn input_globals(&self) -> &[&str] {
        self.inner.borrow_vm().input_global_names()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_arithmetic() {
        let mut program = compile("x := 1 + 2", &[]).unwrap();
        program.run().unwrap();
        assert_eq!(program.global("x").unwrap().as_number(), Some(3.0));
        assert_eq!(program.global("x").unwrap().as_int(), Some(3));
    }

    #[test]
    fn run_leaves_input_globals_null() {
        let mut program = compile("y := x", &["x"]).unwrap();
        program.run().unwrap();
        assert!(program.global("y").unwrap().is_null());
        assert!(program.global("x").unwrap().is_null());
    }

    #[test]
    fn run_with_globals_supplies_inputs() {
        let mut program = compile("y := x + 1", &["x"]).unwrap();
        program
            .run_with_globals(&[("x", Value::number(10.0))])
            .unwrap();
        assert_eq!(program.global("y").unwrap().as_int(), Some(11));
    }

    #[test]
    fn vm_is_reusable_across_runs() {
        let mut program = compile("y := x * 2", &["x"]).unwrap();

        program
            .run_with_globals(&[("x", Value::number(3.0))])
            .unwrap();
        assert_eq!(program.global("y").unwrap().as_int(), Some(6));

        program
            .run_with_globals(&[("x", Value::number(21.0))])
            .unwrap();
        assert_eq!(program.global("y").unwrap().as_int(), Some(42));

        // A plain run resets x to null, so the multiplication fails.
        assert!(program.run().is_err());
    }

    #[test]
    fn inspect_all_globals() {
        let mut program = compile("a := 1\nb := 2\nc := a + b", &[]).unwrap();
        program.run().unwrap();

        let names: Vec<&str> = program.globals().map(|(n, _)| n).collect();
        assert_eq!(names, ["a", "b", "c"]);
        assert_eq!(program.global("c").unwrap().as_int(), Some(3));
    }

    #[test]
    fn string_globals() {
        let mut program = compile("greeting := name + \"!\"", &["name"]).unwrap();
        program
            .run_with_globals(&[("name", Value::string("world".to_string()))])
            .unwrap();
        match program.global("greeting").unwrap().as_value_ref() {
            ValueRef::String(s) => assert_eq!(s, "world!"),
            _ => panic!("expected greeting to be a string"),
        }
    }

    #[test]
    fn functions_and_loops() {
        let source = "\
fn square(n) {
    return n * n
}

total := 0
for i in range(1, 4) {
    total = total + square(i)
}";
        let mut program = compile(source, &[]).unwrap();
        program.run().unwrap();
        // 1 + 4 + 9 = 14
        assert_eq!(program.global("total").unwrap().as_int(), Some(14));
    }

    #[test]
    fn setting_unknown_global_errors() {
        let mut program = compile("x := 1", &[]).unwrap();
        let err = program
            .run_with_globals(&[("nope", Value::number(1.0))])
            .unwrap_err();
        assert!(err.contains("nope"));
    }

    #[test]
    fn setting_script_declared_global_is_rejected() {
        // `x` is script-declared, not an input, so it can't be set even though it exists.
        let mut program = compile("x := 1", &[]).unwrap();
        assert!(
            program
                .run_with_globals(&[("x", Value::number(99.0))])
                .is_err()
        );
        program.run().unwrap();
        assert_eq!(program.global("x").unwrap().as_int(), Some(1));
    }

    #[test]
    fn input_globals_lists_only_inputs() {
        // `a` is an input; `b` is script-declared and must not appear.
        let program = compile("b := a + 1", &["a"]).unwrap();
        assert_eq!(program.input_globals(), ["a"]);
    }

    #[test]
    fn parse_error_is_reported() {
        assert!(compile("x := ", &[]).is_err());
    }

    #[test]
    fn runtime_error_is_reported() {
        // Adding null (the default for an unsupplied input) to a number fails.
        let mut program = compile("y := x + 1", &["x"]).unwrap();
        assert!(program.run().is_err());
    }
}
