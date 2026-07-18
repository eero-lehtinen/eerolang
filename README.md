# Eerolang

Eerolang is dynamically typed programming language implemented in Rust with only a few dependencies (for terminal interaction and logging). It contains a tokenizer, a predictive recursive-descent parser, a one-pass bytecode compiler and a stack based virtual machine (similar to JVM/Python).

## Installation

```sh
cargo install --git https://github.com/eero-lehtinen/eerolang
```

Usage:

```sh
eerolang <file.eel>
```

The instructions can also be stepped through (it shows even more info in a debug build):

```sh
eerolang <file.eel> --step
```

## Use as a library

Eerolang can also be embedded in a Rust program to run scripts and exchange data
with them. Add it as a git dependency:

```sh
cargo add eerolang --git https://github.com/eero-lehtinen/eerolang
```

Then use the API defined in [`src/lib.rs`](src/lib.rs): `compile` a script into a
reusable `Program`, optionally pre-declaring input globals, then `run` it and read
back any global by name.

```rust
use eerolang::{compile, Value};

// Pre-declare `x` as an input global the script can read.
let mut program = compile("result := x * 2", &["x"]).unwrap();

// Supply inputs and run, the program is reusable across runs.
program.run_with_globals(&[("x", Value::number(21.0))]).unwrap();
assert_eq!(program.global("result").unwrap().as_int(), Some(42));
```

Compile errors, parse errors, and runtime errors are all returned as `Err(String)`.

## Motivation

I couldn't think of a language to try this year for the [Advent of Code 2025](https://adventofcode.com/2025) challenge, so I decided to make my own. Check out [my solutions](https://github.com/eero-lehtinen/advent-of-code-2025) if you want. Also I've never made a language before.

## Features

Eerolang is pretty close to Python. It has strong typing, e.g. you can't add a string to a number. It also has dynamic typing, so types are only checked at run time and not when compiling. A list or a map can contain any type in any position.

The types are: `null`, `number`, `string`, `range`, `list`, `queue`, `map`, and `function`. There are no user-defined types. There are also no booleans, all types can be falsy (`null`, `0` for numbers, empty strings, lists, maps). If you really want booleans, I guess you can do `true := 1` and `false := 0` at the start of the file.

The syntax is a combination of Lua, Go and Rust. Declarations use the walrus operator from Go. For-loops and logical expressions are from Lua. Fn and bracing styles are from Rust. The ad hoc grammar also turned out to not need newlines or semicolons at all. Multiple statements in the same file are perfectly fine as long as they are separated by any whitespace (e.g. `x := 10 print(x)`).

The bitwise builtins follow JavaScript semantics: operands are converted to
32-bit integers, shift counts use their low five bits, and results are signed
32-bit integers. `shift_right_unsigned` corresponds to JavaScript's `>>>` and
returns an unsigned value from `0` to `4294967295`.

I spent way too much time making the error messages good, so enjoy that, me.

Below is an example showing off the features of the language:

```python
# Declaration and assignment
x := 10

# Only assignment
x = x + 10

# Numeric operations
x = 10 + 5 * 2 - 3 / 1

# Printing
print("The value of x is", x)

# Looping
for i in range(10) {
    print("i is", i)
    if i > 4 {
        break
    }
    x = x + i
}

# Lists (can contain any type)
list := list(list(1), "asd", 3, list(1, 2, 3))

# Iterating over a list
for i, val in list {
    print("Index:", i, "Value:", val)
}

# Functions (with recursion)
fn fibonacci(x) {
    if x <= 1 {
        return x
    }
    return fibonacci(x - 1) + fibonacci(x - 2)
}

print("Fibonacci of 6 is", fibonacci(6))

# Functions are first-class values (can be stored, passed, and called indirectly)
fn ascending(a, b) { return a - b }
cmp := ascending
print("cmp(3, 1) is", cmp(3, 1))

# Maps (strings as keys and any type as values)
map := map(list("one", 1), list("two", 2), list("three", 3))

# Iterating over a map
for key, value in map {
    print("Key:", key, "Value:", value)
}

# Logical expressions
if 1 and 0 {
    print("This won't print")
}
if 1 or 0 {
    print("This will print")
}
# Not is a function
if not(0) {
    print("Not of any falsy value is 1")
}
# Ternaries also possible (0 is falsy and so are empty strings, lists and maps)
res := x > 10 and "big" or "small"
print("x is", res)

# Builtins:

# Prints any number of arguments of any type
print("Different values:", x, list, map)

# Sleeps for given milliseconds
sleep(10)

# Reads a file and returns its content as a string
file := readfile("samples/features.eel")
# Reads a file without decoding and returns a list of byte values
data := readbytes("image.png")
# Trim whitespace from both ends of a string
trimmed := trim(file)
# Splits a string by a delimiter and returns a list of substrings
lines := split(file, "\n")

# Converts between types
x = int("123")
x = float("3.14")
x = string(456)

# Gets a substring from the first index to the second (exclusive) index
# Can also be given only the last index and the first index defaults to 0
x = substr(file, 0, 3)

# Push an element to the end of a list
push(list, "new element")
# Pops an element from the end of a list and returns it
elem := pop(list)

# There is also a queue, for an efficient pop_front
queue := queue()
push(queue, 1)
push(queue, 2)
first := pop_front(queue)

# Sets a value at index or key in a list or map
list[0] = "changed"
map["one"] = 11

# Gets a value at index or key from a list or map
elem = list[0]
elem = map["one"]

# Chained indexing for nested structures
matrix := list(list(1, 2), list(3, 4))
elem = matrix[0][1]
matrix[1][0] = 99

# Checks if a map contains a key
exists := has(map, "three")

# Removes a value at index or key from a list or map
remove(list, 2)
remove(map, "two")

# Clones a list or map
clonedList := clone(list)

# Gets the length of a string, list or map
length := len(list)

# Math
x = mod(10, 3)
x = bit_and(240, 204)
x = bit_or(240, 15)
x = bit_xor(170, 255)
x = bit_not(0)
x = shift_left(1, 8)
x = shift_right(256, 8)
x = shift_right_unsigned(0 - 1, 1)
x = pow(2, 3)
x = sqrt(2)
x = min(1, 2)
x = max(1, 2)

# Script arguments following `--` in the command line
args := args()
print("Script arguments:", args)
```

## Miri

The VM uses `unsafe` for tagged pointers (SMI encoding) and garbage collection via `transmute`, so [Miri](https://github.com/rust-lang/miri) is useful for checking these for undefined behavior:

```sh
# Run unit tests under Miri
cargo +nightly miri test

# Run the features sample under Miri (very slow)
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri run -- samples/features.eel
```

## Performance notes

Eerolang is not optimized for performance as it is about as slow as Python (not including startup time, which is almost instant with Eerolang). All types are heap allocated and garbage collected by default, except from small integers (SMIs), which are implemented as tagged pointers the same as V8. A lot of extra instructions are generated as there are no optimization passes.

### Performance improvements that are out of scope considering the purposes of the language:

- Interning for strings
- String slices instead of copies
- Optimization passes (might require more intermediate representations between AST and bytecode)
- JIT to machine code
