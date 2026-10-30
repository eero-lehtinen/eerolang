(function_definition
  body: (_) @context.end) @context

(if_statement
  consequence: (_) @context.end) @context

(for_statement
  body: (_) @context.end) @context

(while_statement
  body: (_) @context.end) @context

(call_expression
  arguments: (_
    (_) @context.end)) @context
