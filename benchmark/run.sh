#!/bin/sh

cargo build --release

for program in loop recursion
do
    echo "Benchmarking $program"
    hyperfine --warmup 3 "./target/release/eerolang benchmark/$program.eel" "python benchmark/$program.py"
done
