(function_definition) @function.outer

(function_definition
  body: (block
    .
    "{"
    _+ @function.inner
    "}"))

(if_statement) @conditional.outer

(if_statement
  condition: (_) @conditional.inner)

(if_statement
  consequence: (block
    .
    "{"
    _+ @conditional.inner
    "}"))

(if_statement
  alternative: (block
    .
    "{"
    _+ @conditional.inner
    "}"))

[
  (for_statement)
  (while_statement)
] @loop.outer

(for_statement
  body: (block
    .
    "{"
    _+ @loop.inner
    "}"))

(while_statement
  body: (block
    .
    "{"
    _+ @loop.inner
    "}"))

(block
  .
  "{"
  _+ @block.inner
  "}") @block.outer

(call_expression) @call.outer

(call_expression
  arguments: (arguments
    .
    "("
    _+ @call.inner
    ")"))

(return_statement
  value: (_) @return.inner) @return.outer

[
  (declaration)
  (assignment)
  (subscript_assignment)
] @assignment.outer

(declaration
  name: (_) @assignment.lhs
  value: (_) @assignment.rhs @assignment.inner)

(assignment
  name: (_) @assignment.lhs
  value: (_) @assignment.rhs @assignment.inner)

(subscript_assignment
  target: (_) @assignment.lhs
  value: (_) @assignment.rhs @assignment.inner)

(comment) @comment.outer

(number) @number.inner

; Parameters and arguments: the first element takes the following comma, the
; rest take the preceding one, so deleting `@parameter.outer` leaves a valid list.
(parameters
  .
  (identifier) @parameter.inner @parameter.outer
  .
  ","? @parameter.outer)

(parameters
  "," @parameter.outer
  .
  (identifier) @parameter.inner @parameter.outer)

(arguments
  .
  (_) @parameter.inner @parameter.outer
  .
  ","? @parameter.outer)

(arguments
  "," @parameter.outer
  .
  (_) @parameter.inner @parameter.outer)
