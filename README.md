# Eerolang

Suitable only for non-serious purposes.

Eerolang is implemented in Rust with only a few dependencies (for terminal interaction and logging). It contains a tokenizer, a predictive recursive-descent parser, a one pass bytecode compiler and a virtual machine.

## Installation

```sh
cargo install --git https://github.com/eero-lehtinen/eerolang
```

Usage:

```sh
eerolang <file.eel>
```

## Motivation

I couldn't think of a language to try this year for the [Advent of Code 2025](https://adventofcode.com/2025) challenge, so I decided to make my own. Check out [my solutions](https://github.com/eero-lehtinen/advent-of-code-2025) if you want.

I've also never tried to make a language before, so I wanted to figure out how that works.

## Features

Eerolang is pretty close to Python. It has strong typing, e.g. you can't add a string to a number. It also has dynamic typing, so types are only checked at run time and not when compiling. Also a list or a map can contain any type in any position.

The types are: `number`, `string`, `range`, `list`, and `map`. There are no user-defined types. There are also no booleans, all types can be falsy (0 for numbers, empty strings, lists, and maps). If you really want booleans, I guess you can do `true := 1` and `false := 0` at the start of the file.

The syntax is a combination of Lua, Go and Rust. Declarations use the walrus operator from Go. For-loops and logical expressions are from Lua. Fn and bracing styles are from Rust. The ad hoc grammar also turned out to not need newlines or semicolons at all. Multiple statements in the same file are perfectly fine as long as they are separated by any whitespace (e.g. `x := 10 print(x)`).

I spent way too much time making the error messages good for no reason, so enjoy that, me.

Below is an exaple showing off the features of the language:

```python
# Declaration and assignment
x := 10

# Only assignment
x = x + 10

# Numeric operations
x = 10 + 5 * 2 - 3 / 1

# Printing
print("The value of x is", x)

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

# Functions
fn fibonacci(x) {
    if x <= 1 {
        return x
    }
    return fibonacci(x - 1) + fibonacci(x - 2)
}

print("Fibonacci of 6 is", fibonacci(6))

# Maps (strings as keys and any type as values)
map := map(list("one", 1), list("two", 2), list("three", 3))

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

# Sets a value at index or key in a list or map
set(list, 0, "changed")
set(map, "one", 11)

# Gets a value at index or key from a list, string or map
elem = get(list, 0)
elem = get("hello", 1)

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
x = pow(2, 3)
x = sqrt(2)
```

## Performance notes

Eerolang is not optimized for performance as it is about as slow as Python (not including startup time, which is almost instant with Eerolang). All types are heap allocated and reference counted by default, except from small integers (SMIs), which are implemented as tagged pointers the same as V8. A lot of extra instructions are generated as there are no optimization passes. The number of instructions could probably be at least halved, likely even more for function heavy programs.

### Performance improvements that are out of scope considering the purposes of the language:

- Carbage collection to replace reference counting
- Interning for strings
- String slices instead of copies
- Optimization passes (might require more intermediate representations between AST and bytecode)
- JIT to machine code
