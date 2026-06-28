//! Behavioral edge-case tests driving the language through the public API.
//! Grouped by the stage most likely to be exercised: tokenizer, parser,
//! compiler, then a few end-to-end runtime checks.

use eerolang::{Value, ValueRef, compile};

/// Compiles + runs `src` (no inputs) and returns the named global.
fn get(src: &str, var: &str) -> Value {
    let mut p = compile(src, &[]).unwrap_or_else(|e| panic!("compile failed: {e}"));
    p.run().unwrap_or_else(|e| panic!("run failed: {e}"));
    p.global(var).unwrap_or_else(|| panic!("no global '{var}'"))
}

fn int(src: &str, var: &str) -> i64 {
    get(src, var)
        .as_int()
        .unwrap_or_else(|| panic!("global '{var}' is not an int"))
}

fn num(src: &str, var: &str) -> f64 {
    get(src, var)
        .as_number()
        .unwrap_or_else(|| panic!("global '{var}' is not a number"))
}

fn string(src: &str, var: &str) -> String {
    let v = get(src, var);
    match v.as_value_ref() {
        ValueRef::String(s) => s.to_string(),
        _ => panic!("global '{var}' is not a string"),
    }
}

/// True if the program fails to compile.
fn compile_err(src: &str) -> bool {
    compile(src, &[]).is_err()
}

/// True if the program compiles but fails at runtime.
fn run_err(src: &str) -> bool {
    match compile(src, &[]) {
        Ok(mut p) => p.run().is_err(),
        Err(_) => panic!("expected a clean compile, but compilation failed"),
    }
}

// ---------------------------------------------------------------- tokenizer

#[test]
fn integer_and_float_literals() {
    assert_eq!(int("x := 42", "x"), 42);
    assert_eq!(num("x := 1.5", "x"), 1.5);
    assert_eq!(num("x := 2.0", "x"), 2.0);
    // Trailing-dot float.
    assert_eq!(num("x := 5.", "x"), 5.0);
}

#[test]
fn tokens_without_surrounding_whitespace() {
    assert_eq!(int("x:=1+2*3", "x"), 7);
    assert_eq!(int("x:=(1+2)*3", "x"), 9);
}

#[test]
fn number_followed_by_letters_is_rejected() {
    assert!(compile_err("x := 123abc"));
}

#[test]
fn string_escapes() {
    assert_eq!(string(r#"x := "a\nb""#, "x"), "a\nb");
    assert_eq!(string(r#"x := "tab\there""#, "x"), "tab\there");
    assert_eq!(string(r#"x := "quote\"end""#, "x"), "quote\"end");
    assert_eq!(string(r#"x := "back\\slash""#, "x"), "back\\slash");
}

#[test]
fn string_concatenation() {
    assert_eq!(string(r#"x := "foo" + "bar""#, "x"), "foobar");
}

#[test]
fn unterminated_string_is_rejected() {
    assert!(compile_err(r#"x := "abc"#));
}

#[test]
fn trailing_comment_without_newline() {
    assert_eq!(int("x := 5 # trailing comment", "x"), 5);
}

#[test]
fn comment_lines_are_ignored() {
    let src = "# leading comment\nx := 1\n# middle\ny := x + 1";
    assert_eq!(int(src, "y"), 2);
}

#[test]
fn carriage_returns_are_whitespace() {
    assert_eq!(int("x := 1\r\ny := x + 1", "y"), 2);
}

#[test]
fn bang_without_equals_is_rejected() {
    assert!(compile_err("x := 1 ! 2"));
}

#[test]
fn unicode_string_literals_round_trip() {
    assert_eq!(string("x := \"héllo wörld\"", "x"), "héllo wörld");
    assert_eq!(string("x := \"café\" + \" ☕\"", "x"), "café ☕");
    assert_eq!(string("x := \"日本語\"", "x"), "日本語");
}

#[test]
fn len_counts_characters_not_bytes() {
    assert_eq!(int("x := len(\"héllo\")", "x"), 5);
    assert_eq!(int("x := len(\"日本語\")", "x"), 3);
    // Emoji outside the BMP are a single scalar value.
    assert_eq!(int("x := len(\"a🎉b\")", "x"), 3);
}

#[test]
fn string_subscript_is_char_indexed() {
    assert_eq!(string("x := \"héllo\"[0]", "x"), "h");
    assert_eq!(string("x := \"héllo\"[1]", "x"), "é");
    assert_eq!(string("x := \"héllo\"[4]", "x"), "o");
    assert_eq!(string("x := \"a🎉b\"[1]", "x"), "🎉");
}

#[test]
fn string_subscript_past_last_char_is_clean_out_of_bounds() {
    // "héllo" has 5 chars (6 bytes). Index 5 must be a clean runtime error,
    // not a panic from indexing past the char count.
    assert!(run_err("x := \"héllo\"[5]"));
}

#[test]
fn substr_is_char_indexed() {
    assert_eq!(string("x := substr(\"héllo\", 1, 3)", "x"), "él");
    // A range whose byte span would split a multi-byte char must not panic.
    assert_eq!(string("x := substr(\"héllo\", 0, 2)", "x"), "hé");
    assert_eq!(string("x := substr(\"a🎉b\", 1, 2)", "x"), "🎉");
}

#[test]
fn unicode_identifiers() {
    // `is_alphabetic` accepts Unicode letters, so these are valid identifiers.
    assert_eq!(int("café := 3\nr := café + 1", "r"), 4);
}

// ------------------------------------------------------------------- parser

#[test]
fn arithmetic_precedence() {
    assert_eq!(int("x := 2 + 3 * 4", "x"), 14);
    assert_eq!(int("x := 2 * 3 + 4", "x"), 10);
    assert_eq!(int("x := (2 + 3) * 4", "x"), 20);
}

#[test]
fn subtraction_and_division_are_left_associative() {
    assert_eq!(int("x := 10 - 2 - 3", "x"), 5);
    assert_eq!(int("x := 20 / 2 / 5", "x"), 2);
}

#[test]
fn unary_minus() {
    assert_eq!(int("x := 0 - 5", "x"), -5);
    assert_eq!(int("x := -5", "x"), -5);
    assert_eq!(int("x := 3 - -2", "x"), 5);
}

#[test]
fn comparison_operators() {
    assert_eq!(int("x := 5 > 3", "x"), 1);
    assert_eq!(int("x := 3 >= 3", "x"), 1);
    assert_eq!(int("x := 2 < 1", "x"), 0);
    assert_eq!(int("x := 2 <= 2", "x"), 1);
    assert_eq!(int("x := 2 == 2", "x"), 1);
    assert_eq!(int("x := 2 != 3", "x"), 1);
}

#[test]
fn logical_operators_and_ternary_idiom() {
    assert_eq!(int("x := 1 and 1", "x"), 1);
    assert_eq!(int("x := 1 and 0", "x"), 0);
    assert_eq!(int("x := 0 or 7", "x"), 7);
    assert_eq!(string(r#"x := 1 and "yes" or "no""#, "x"), "yes");
    assert_eq!(string(r#"x := 0 and "yes" or "no""#, "x"), "no");
}

#[test]
fn logical_operators_short_circuit() {
    // If `or`/`and` did not short-circuit, the `1 / 0` would trigger a
    // division-by-zero fatal error.
    assert_eq!(int("x := 1 or (1 / 0)", "x"), 1);
    assert_eq!(int("x := 0 and (1 / 0)", "x"), 0);
}

#[test]
fn if_else_chooses_branch() {
    assert_eq!(int("x := 0\nif 1 { x = 1 } else { x = 2 }", "x"), 1);
    assert_eq!(int("x := 0\nif 0 { x = 1 } else { x = 2 }", "x"), 2);
}

#[test]
fn else_if_chain_is_not_supported() {
    // `else if` requires a block after `else`; this documents the limitation.
    assert!(compile_err("x := 0\nif 0 { x = 1 } else if 1 { x = 2 }"));
    // The nested-block workaround does work:
    let src = "x := 0\nif 0 { x = 1 } else { if 1 { x = 2 } else { x = 3 } }";
    assert_eq!(int(src, "x"), 2);
}

#[test]
fn empty_and_comment_only_programs_are_valid_no_ops() {
    for src in ["", "   \n  \t ", "# only a comment\n"] {
        let mut p = compile(src, &[]).unwrap_or_else(|e| panic!("compile failed for {src:?}: {e}"));
        p.run()
            .unwrap_or_else(|e| panic!("run failed for {src:?}: {e}"));
        assert!(
            p.globals().next().is_none(),
            "expected no globals for {src:?}"
        );
    }
}

#[test]
fn deeply_nested_parentheses() {
    assert_eq!(int("x := ((((1 + 2))))", "x"), 3);
}

// ----------------------------------------------------------------- compiler

#[test]
fn reassignment_after_declaration() {
    assert_eq!(int("x := 1\nx = x + 4", "x"), 5);
}

#[test]
fn assigning_undeclared_variable_is_rejected() {
    assert!(compile_err("x = 1"));
}

#[test]
fn reading_undeclared_variable_is_rejected() {
    assert!(compile_err("y := z"));
}

#[test]
fn duplicate_declaration_in_same_scope_is_rejected() {
    assert!(compile_err("x := 1\nx := 2"));
}

#[test]
fn block_scoped_shadowing() {
    // A declaration inside the block shadows the global; the global is intact
    // afterwards.
    let src = "outer := 1\nshadowed := 0\nif 1 { outer := 2\nshadowed = outer }\nfinal := outer";
    assert_eq!(int(src, "shadowed"), 2);
    assert_eq!(int(src, "final"), 1);
}

#[test]
fn redefining_builtin_is_rejected() {
    assert!(compile_err("fn len(x) { return 0 }"));
}

#[test]
fn redefining_function_is_rejected() {
    assert!(compile_err("fn f() { return 1 }\nfn f() { return 2 }"));
}

#[test]
fn wrong_argument_count_is_rejected() {
    assert!(compile_err("fn f(a) { return a }\nr := f(1, 2)"));
    assert!(compile_err("fn f(a, b) { return a }\nr := f(1)"));
}

#[test]
fn duplicate_parameter_names_are_rejected() {
    assert!(compile_err("fn f(a, a) { return a }\nr := f(1, 2)"));
}

#[test]
fn function_can_read_globals() {
    let src = "g := 10\nfn addg(n) { return n + g }\nr := addg(5)";
    assert_eq!(int(src, "r"), 15);
}

#[test]
fn function_without_return_yields_null() {
    let mut p = compile("fn f() { }\nr := f()", &[]).unwrap();
    p.run().unwrap();
    assert!(p.global("r").unwrap().is_null());
}

// ------------------------------------------------------------------ runtime

#[test]
fn recursion() {
    let src = "fn fib(n) { if n <= 1 { return n }\nreturn fib(n - 1) + fib(n - 2) }\nr := fib(10)";
    assert_eq!(int(src, "r"), 55);
}

#[test]
fn function_value_reference() {
    assert_eq!(
        int("fn sq(n) { return n * n }\nf := sq\nr := f(6)", "r"),
        36
    );
}

#[test]
fn builtin_value_reference() {
    assert_eq!(int("f := len\nr := f(\"abcd\")", "r"), 4);
}

#[test]
fn while_loop_accumulates() {
    let src = "i := 0\ns := 0\nwhile i < 5 { s = s + i\ni = i + 1 }\nr := s";
    assert_eq!(int(src, "r"), 10);
}

#[test]
fn for_range_loops() {
    assert_eq!(
        int("s := 0\nfor i in range(5) { s = s + i }\nr := s", "r"),
        10
    );
    assert_eq!(
        int("s := 0\nfor i in range(2, 5) { s = s + i }\nr := s", "r"),
        9
    );
}

#[test]
fn for_break_and_continue() {
    let brk = "s := 0\nfor i in range(10) { if i >= 3 { break }\ns = s + i }\nr := s";
    assert_eq!(int(brk, "r"), 3);
    let cont = "s := 0\nfor i in range(5) { if mod(i, 2) == 0 { continue }\ns = s + i }\nr := s";
    assert_eq!(int(cont, "r"), 4);
}

#[test]
fn nested_loops() {
    let src = "s := 0\nfor i in range(3) { for j in range(3) { s = s + 1 } }\nr := s";
    assert_eq!(int(src, "r"), 9);
}

#[test]
fn list_indexing_and_assignment() {
    assert_eq!(int("l := list(10, 20, 30)\nr := l[1]", "r"), 20);
    assert_eq!(int("l := list(1, 2, 3)\nl[0] = 99\nr := l[0]", "r"), 99);
}

#[test]
fn chained_subscript() {
    let src = "m := list(list(1, 2), list(3, 4))\nr := m[1][0]";
    assert_eq!(int(src, "r"), 3);
    let src2 = "m := list(list(1, 2), list(3, 4))\nm[0][1] = 77\nr := m[0][1]";
    assert_eq!(int(src2, "r"), 77);
}

#[test]
fn map_access() {
    let src = "m := map(list(\"k\", 5), list(\"j\", 6))\nr := m[\"k\"]";
    assert_eq!(int(src, "r"), 5);
}

#[test]
fn math_builtins() {
    assert_eq!(int("r := pow(2, 10)", "r"), 1024);
    assert_eq!(int("r := mod(17, 5)", "r"), 2);
    assert_eq!(int("r := min(3, 7)", "r"), 3);
    assert_eq!(int("r := max(3, 7)", "r"), 7);
    assert_eq!(num("r := sqrt(16)", "r"), 4.0);
}

#[test]
fn type_conversions() {
    assert_eq!(int(r#"r := int("42")"#, "r"), 42);
    assert_eq!(num(r#"r := float("2.5")"#, "r"), 2.5);
    assert_eq!(string("r := string(7)", "r"), "7");
}

#[test]
fn string_builtins() {
    assert_eq!(int(r#"r := len("hello")"#, "r"), 5);
    assert_eq!(string(r#"r := substr("hello", 0, 3)"#, "r"), "hel");
    assert_eq!(string(r#"r := trim("  hi  ")"#, "r"), "hi");
}

#[test]
fn division_by_zero_is_a_runtime_error() {
    assert!(run_err("r := 1 / 0"));
}

#[test]
fn type_mismatch_is_a_runtime_error() {
    assert!(run_err(r#"r := "a" - 1"#));
}
