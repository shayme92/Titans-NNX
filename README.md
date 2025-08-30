# Titans-NNX: A JAX + Flax NNX Implementation of *“Titans: Learning to Memorize at Test Time”*

This repository contains an **unofficial**, open-source implementation of the paper *“Titans: Learning to Memorize at Test Time”* in JAX and Flax NNX. The project aims to provide a clear, well-documented, professional codebase for researchers and developers interested in models that combine short-term attention with long-term, test-time-updatable memory.

## Paper

**Titans: Learning to Memorize at Test Time**  
*Paper:* [https://arxiv.org/pdf/2501.00663](https://arxiv.org/pdf/2501.00663)

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/shayme92/Titans-NNX.git
cd Titans-NNX
pip install -r requirements.txt
```

## Usage

### Training

Run:

```bash
python training/train_mac.py --config configs/mac.yaml --train_path /path/to/train/data --val_path /path/to/val/data
```

You can customize training by modifying `configs/mac.yaml` or by passing command-line arguments.

### Text Generation

You can add a generation script or extend the existing training script. An example of how to use the `generate_text` method is in `variants/mac.py`.

## Architecture Implementation Details

### Parallelization

The model’s primary scaling bottleneck is the forward update of the long-term memory for each sequence (associative scan). We implement the paper’s gradient-descent-with-momentum update. Sequences in a batch are parallelized with `vmap` and a device mesh for efficient processing.

### Long-Term Memory

Memory parameters are reinitialized at the start of each sequence and optimized in a feed-forward manner during token generation. Training mimics this test-time behavior. The long-term memory parameters are **not** updated via backpropagation, preserving the test-time optimization dynamics.

## Architecture Assumptions

Because the paper does not provide a complete, code-level specification, some components are implemented based on reasonable assumptions:

- **Q, K, V projections:** Dimensions and linearity of the query, key, and value projections in the long-term memory module.
- **Gated nonlinearity:** Implementation details of the gated nonlinearity used in the final layers.
- **MLP architecture:** Size and structure of the multi-layer perceptron (MLP) components within the long-term memory system.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
