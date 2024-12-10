Work in Progress. I am still building the elements I need...

# GPT Implementation

A PyTorch implementation of a GPT (Generative Pre-trained Transformer) model with Byte-Pair Encoding (BPE) tokenization. This implementation is designed to be educational and extensible, allowing for training on custom text datasets.

## Features

- Transformer-based architecture with multi-head self-attention
- Byte-Pair Encoding (BPE) tokenization
- Configurable model parameters
- Training and inference capabilities
- Support for custom text datasets

## Project Structure 

gpt/
├── bpe.py # BPE tokenizer implementation
├── config.py # Configuration settings
├── data.py # Dataset handling and preprocessing
├── model.py # GPT model architecture
├── train.py # Training and inference logic
└── input.txt # Your training data

## Requirements

- Python 3.8+
- PyTorch 1.8+
- typing
- collections

## Quick Start

1. Place your training text in `input.txt`

2. Configure the model in `config.py`:

```python
python
class Config:
batch_size = 64 # batch size for training
block_size = 128 # context window size
n_embd = 384 # embedding dimension
n_head = 6 # number of attention heads
n_layer = 6 # number of transformer blocks
vocab_size = 512 # vocabulary size for BPE
# ... other parameters
```

3. Run `python train.py` to train the model.

## Components

### BPE Tokenizer (bpe.py)
Implements Byte-Pair Encoding for tokenization, starting with byte-level tokens and merging common pairs to build a vocabulary.

### Dataset Handler (data.py)
Manages data loading, tokenization, and batch generation for training.

### Model Architecture (model.py)
Implements the GPT architecture with:
- Token and position embeddings
- Multi-head self-attention
- Feed-forward networks
- Layer normalization

### Training (train.py)
Handles:
- Model initialization
- Training loop
- Loss calculation
- Model saving/loading
- Text generation

## Usage

### Training
The model will train on your input text and save the trained model to `model.pth`.

### Text Generation
After training, you can generate text by providing a prompt:

```python
question = "What is Monstera?"
generated_text = model.generate(encoded_question, max_new_tokens=500)
```

## Customization

- Adjust model size and training parameters in `config.py`
- Modify the tokenization approach in `bpe.py`
- Extend the model architecture in `model.py`
- Customize the training loop in `train.py`

## License

MIT License

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.