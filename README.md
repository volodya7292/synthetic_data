# synthetic_data

Implements synthetic data generation using PSVAE.

## Building

### Prerequisites

Rust (https://www.rust-lang.org)

#### Windows/Linux
1. Download PyTorch: [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

#### macOS (Arm)

1. Download PyTorch using pip/conda: [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

Set `LIBTORCH=<pytorch folder>` environment variable.

### Build

Run `cargo build --release`. The library is generated in `/target/release/` directory.
 
