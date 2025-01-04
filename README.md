# synthetic_data

Implements synthetic data generation using PSVAE model from https://arxiv.org/abs/2407.13016.

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
 
### Citation

```bibtex
@misc{highquality_tabular_datagen_2024,
      title={High-Quality Tabular Data Generation using Post-Selected VAE}, 
      author={Volodymyr Shulakov},
      year={2024},
      eprint={2407.13016},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.13016}, 
}
```
