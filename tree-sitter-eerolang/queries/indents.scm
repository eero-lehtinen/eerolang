[
  (block)
  (arguments)
  (parameters)
  (parenthesized_expression)
] @indent.begin

(block
  "}" @indent.branch)

(arguments
  ")" @indent.branch)

(parameters
  ")" @indent.branch)

(parenthesized_expression
  ")" @indent.branch)

[
  "}"
  ")"
] @indent.end

(string) @indent.auto

(comment) @indent.ignore
