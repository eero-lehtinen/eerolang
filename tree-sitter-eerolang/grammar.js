/**
 * @file Eerolang grammar for tree-sitter
 * @author Eero Lehtinen
 * @license MIT
 */

/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

// Mirrors `Operator::precedence` in src/tokenizer.rs, shifted up by one.
const PREC = {
  or: 1,
  and: 2,
  compare: 3,
  additive: 4,
  multiplicative: 5,
  unary: 6,
  postfix: 7,
};

export default grammar({
  name: "eerolang",

  extras: ($) => [/\s/, $.comment],

  word: ($) => $.identifier,

  supertypes: ($) => [$._statement, $._expression],

  rules: {
    // Functions can only be defined at the top level.
    source_file: ($) => repeat(choice($.function_definition, $._statement)),

    _statement: ($) =>
      choice(
        $.declaration,
        $.assignment,
        $.subscript_assignment,
        // A call at the start of a statement is a complete statement, e.g.
        // `f(x) - 1` is a syntax error, like in src/ast_parser.rs.
        prec(1, $.call_expression),
        $.return_statement,
        $.if_statement,
        $.for_statement,
        $.while_statement,
        $.break_statement,
        $.continue_statement,
      ),

    function_definition: ($) =>
      seq(
        "fn",
        field("name", $.identifier),
        field("parameters", $.parameters),
        field("body", $.block),
      ),

    parameters: ($) => seq("(", commaSep($.identifier), ")"),

    block: ($) => seq("{", repeat($._statement), "}"),

    declaration: ($) =>
      seq(field("name", $.identifier), ":=", field("value", $._expression)),

    assignment: ($) =>
      seq(field("name", $.identifier), "=", field("value", $._expression)),

    subscript_assignment: ($) =>
      seq(
        field("target", $.subscript_expression),
        "=",
        field("value", $._expression),
      ),

    return_statement: ($) => seq("return", field("value", $._expression)),

    // There is no `else if`, `else` must be followed by a block.
    if_statement: ($) =>
      seq(
        "if",
        field("condition", $._expression),
        field("consequence", $.block),
        optional(seq("else", field("alternative", $.block))),
      ),

    for_statement: ($) =>
      seq(
        "for",
        field("key", $.identifier),
        optional(seq(",", field("value", $.identifier))),
        "in",
        field("iterable", $._expression),
        field("body", $.block),
      ),

    while_statement: ($) =>
      seq("while", field("condition", $._expression), field("body", $.block)),

    break_statement: (_) => "break",

    continue_statement: (_) => "continue",

    _expression: ($) =>
      choice(
        $.identifier,
        $.number,
        $.string,
        $.null,
        $.call_expression,
        $.subscript_expression,
        $.unary_expression,
        $.binary_expression,
        $.parenthesized_expression,
      ),

    // Only plain identifiers are callable, e.g. `list[0](x)` is not valid.
    call_expression: ($) =>
      prec(
        PREC.postfix,
        seq(field("function", $.identifier), field("arguments", $.arguments)),
      ),

    arguments: ($) => seq("(", commaSep($._expression), ")"),

    subscript_expression: ($) =>
      prec(
        PREC.postfix,
        seq(
          field("object", $._expression),
          "[",
          field("index", $._expression),
          "]",
        ),
      ),

    unary_expression: ($) =>
      prec(PREC.unary, seq(field("operator", "-"), field("operand", $._expression))),

    binary_expression: ($) => {
      const table = [
        [PREC.or, "or"],
        [PREC.and, "and"],
        [PREC.compare, choice("<", ">", "<=", ">=", "==", "!=")],
        [PREC.additive, choice("+", "-")],
        [PREC.multiplicative, choice("*", "/")],
      ];
      return choice(
        ...table.map(([precedence, operator]) =>
          prec.left(
            /** @type {number} */ (precedence),
            seq(
              field("left", $._expression),
              // @ts-ignore
              field("operator", operator),
              field("right", $._expression),
            ),
          ),
        ),
      );
    },

    parenthesized_expression: ($) => seq("(", $._expression, ")"),

    identifier: (_) => /[\p{L}_][\p{L}\p{N}_]*/,

    number: (_) => /\d+(\.\d*)?/,

    // Strings may span multiple lines.
    string: ($) =>
      seq(
        '"',
        repeat(choice($.string_content, $.escape_sequence)),
        token.immediate('"'),
      ),

    string_content: (_) => token.immediate(prec(1, /[^"\\]+/)),

    escape_sequence: (_) => token.immediate(/\\(.|\n)/),

    null: (_) => "null",

    comment: (_) => token(seq("#", /.*/)),
  },
});

/**
 * Comma separated list that allows a trailing comma.
 *
 * @param {RuleOrLiteral} rule
 */
function commaSep(rule) {
  return optional(seq(rule, repeat(seq(",", rule)), optional(",")));
}
