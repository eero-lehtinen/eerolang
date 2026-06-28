#!/usr/bin/env pwsh

cargo build --release

foreach ($program in "loop", "recursion") {
    Write-Host "Benchmarking $program"
    hyperfine --warmup 3 ".\target\release\eerolang.exe benchmark\$program.eel" "python benchmark\$program.py"
}
