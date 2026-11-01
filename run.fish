#!/usr/bin/env fish
set -l dir (dirname (status filename))
cargo run --release --manifest-path $dir/Cargo.toml -- $argv
