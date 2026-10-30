; Later patterns take priority in Neovim, so general captures come first.

(identifier) @variable

(parameters
  (identifier) @variable.parameter)

(function_definition
  name: (identifier) @function)

(call_expression
  function: (identifier) @function.call)

(call_expression
  function: (identifier) @function.builtin
  (#any-of? @function.builtin
    "list" "queue" "map" "range" "args" "not" "print" "sleep" "readfile" "readbytes" "trim"
    "split" "int" "float" "string" "substr" "push" "pop" "pop_front" "set" "get" "has" "remove"
    "clone" "len" "mod" "bit_and" "bit_or" "bit_xor" "bit_not" "shift_left" "shift_right"
    "shift_right_unsigned" "pow" "sqrt" "min" "max"))

; `_` discards the key or value in a for loop.
(for_statement
  [
    key: (identifier)
    value: (identifier)
  ] @variable.builtin
  (#eq? @variable.builtin "_"))

"fn" @keyword.function

"return" @keyword.return

[
  "if"
  "else"
] @keyword.conditional

[
  "for"
  "while"
  "in"
  (break_statement)
  (continue_statement)
] @keyword.repeat

[
  "and"
  "or"
] @keyword.operator

[
  ":="
  "="
  "+"
  "-"
  "*"
  "/"
  "<"
  ">"
  "<="
  ">="
  "=="
  "!="
] @operator

[
  "("
  ")"
  "["
  "]"
  "{"
  "}"
] @punctuation.bracket

"," @punctuation.delimiter

(number) @number

((number) @number.float
  (#lua-match? @number.float "%."))

(string) @string

(escape_sequence) @string.escape

(null) @constant.builtin

(comment) @comment @spell
