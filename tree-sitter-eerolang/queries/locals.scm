[
  (source_file)
  (function_definition)
  (block)
  (for_statement)
] @local.scope

(function_definition
  name: (identifier) @local.definition.function)

(parameters
  (identifier) @local.definition.parameter)

(declaration
  name: (identifier) @local.definition.var)

(for_statement
  key: (identifier) @local.definition.var)

(for_statement
  value: (identifier) @local.definition.var)

(identifier) @local.reference
