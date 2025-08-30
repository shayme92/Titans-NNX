# Titans-NNX: A JAX + NNX Implementation of "Titans: Learning to Memorize at Test Time"

This repository contains a nonofficial open-source implementation of the paper "Titans: Learning to Memorize at Test Time" in JAX and Flax's NNX. The project aims to provide a clear, well-documented, and professional codebase for researchers and developers interested in models that combine short-term attention with long-term, test-time updatable memory.

## Paper

**Titans: Learning to Memorize at Test Time**  
*Paper Link*: [https://arxiv.org/pdf/2501.00663](https://arxiv.org/pdf/2501.00663)

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/shayme92/Titans-NNX.git
cd Titans-NNX
pip install -r requirements.txt
```

## Usage

### Training

To train the model, run the following command:

```bash
python training/train_mac.py --config configs/mac.yaml
```

You can customize the training process by modifying the `configs/mac.yaml` file or by providing command-line arguments.

### Text Generation

To generate text with a trained model, you can add a generation script or extend the existing training script. An example of how to use the `generate_text` method can be found in `variants/mac.py`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
