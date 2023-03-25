# synthetic_data

A dynamic system library for synthetic data generation using
slightly simplified [TVAE](https://arxiv.org/abs/1907.00503)
architecture (for continuous data uses min-max normalization instead of mode-specific).

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
 
