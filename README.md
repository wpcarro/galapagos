# Galapagos

[![Crates.io](https://img.shields.io/crates/v/galapagos.svg)](https://crates.io/crates/galapagos)
[![Docs.rs](https://docs.rs/galapagos/badge.svg)](https://docs.rs/galapagos)
[![License: Unlicensed](https://img.shields.io/badge/License-Unlicensed-blue.svg)](https://unlicense.org/)

Simple evolutionary solver written in Rust.

<img src="images/history.png" alt="Graph of fitness over time" width="500" />

## Usage

```rust
use matrix_rs::galapagos::{self, Goal};

fn main() {
    let solution = galapagos::solve(
        |xs| {
            // Rosenbrock: (a - x)^2 + b(y - x^2)^2
            let x = xs[0];
            let y = xs[1];
            (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2)
        },
        &[(-5.0, 5.0), (-5.0, 5.0)],
        Goal::Minimize,
        Default::default(),
    );
    println!("{solution:?}");
}
```