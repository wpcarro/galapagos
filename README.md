# Galapagos

Simple evolutionary solver written in Rust.

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
        vec![(-5.0, 5.0), (-5.0, 5.0)],
        Goal::Minimize,
        Default::default(),
    );
    println!("{solution:?}");
}
```