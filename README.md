# GPT Language Model

A PyTorch implementation of GPT (Generative Pre-trained Transformer) models with modern architecture improvements and a clean, simple interface.

## Quick Start

### Setup
```bash
git clone https://github.com/tmickleydoyle/language-model.git
cd language-model
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Basic Usage

#### 1. Train a Model
```bash
# Train on a single file
python train.py example/raw/science_article_01.txt -o my_model

# Train on all files in directory  
python train.py example/raw/ -o comprehensive_model
```

#### 2. Fine-tune on Q&A Data  
```bash
python fine-tune.py my_model example/fine-tuned/story_qa_dataset.json -o fine_tuned_model
```

#### 3. Ask Questions
```bash
python ask.py fine_tuned_model --interactive
```

## Model Architecture

This implementation includes modern transformer improvements:
- **Grouped Query Attention**: More efficient than standard multi-head attention
- **RMSNorm**: Better normalization than LayerNorm
- **RoPE**: Rotary positional encoding for better position understanding
- **xSwiGLU**: Advanced activation function
- **QK-Norm**: Improved training stability
- **Flash Attention**: Optimized attention computation

## Configuration Options

You can customize the model architecture by modifying the config:

```python
from src.config import Config

# Standard configuration
config = Config(
    n_embd=512,        # Embedding dimension
    n_head=8,          # Number of attention heads  
    n_layer=6,         # Number of transformer layers
    block_size=1024,   # Context window size
    vocab_size=32000   # Vocabulary size
)

# Advanced options
config.attention_type = 'mla'     # Use Multi-Head Latent Attention (memory efficient)
config.attention_type = 'swa'     # Use Sliding Window Attention (long sequences)
config.position_encoding_type = 'cope'  # Use Contextual Position Encoding
config.use_moe = True             # Use Mixture of Experts (scalable capacity)
```

## Project Structure

```
language-model/
├── src/                     # Source code modules
│   ├── config.py            # Configuration management
│   ├── model/               # GPT model implementation
│   ├── tokenizer/           # BPE tokenizer
│   ├── data/                # Dataset handling
│   ├── training/            # Training infrastructure
│   └── utils/               # Utility functions
├── tests/                   # Test suite
├── example/                 # Training datasets
│   ├── raw/                 # Raw text for training
│   └── fine-tuned/          # Q&A datasets
├── train.py                 # Main training script
├── fine-tune.py             # Fine-tuning script
├── ask.py                   # Interactive Q&A script
└── README.md                # This file
```

## Usage Guide

### Training a Model

Train a GPT model from scratch:

```bash
# Basic training (uses modern architecture by default)
python train.py example/raw/science_article_01.txt -o my_model

# Custom model configuration
python train.py example/raw/ \
    --embedding-dim 512 \
    --num-heads 8 \
    --num-layers 6 \
    --context-size 1024 \
    --batch-size 16 \
    --epochs 10 \
    -o custom_model

# Memory-efficient training (smaller model)
python train.py example/raw/ \
    --embedding-dim 256 \
    --num-heads 8 \
    --num-layers 4 \
    --context-size 512 \
    --batch-size 32 \
    --epochs 5 \
    -o small_model
```

**Training Options:**
- `--embedding-dim`: Model dimension (default: 128)
- `--num-heads`: Number of attention heads (default: 4)  
- `--num-layers`: Number of transformer layers (default: 3)
- `--context-size`: Context window size (default: 128)
- `--epochs`: Training epochs (default: 50)
- `--batch-size`: Batch size (default: 16)
- `--learning-rate`: Learning rate (default: 5e-4)
- `--dropout`: Dropout rate (default: 0.3)
- `-o, --output`: Output directory name

### Fine-tuning a Model

Fine-tune a trained model on Q&A data:

```bash
# Basic fine-tuning
python fine-tune.py my_model example/fine-tuned/story_qa_dataset.json -o fine_tuned_model

# Custom fine-tuning parameters
python fine-tune.py my_model example/fine-tuned/story_qa_dataset.json \
    --epochs 10 \
    --batch-size 8 \
    --learning-rate 1e-5 \
    -o specialized_model
```

**Fine-tuning Options:**
- `--epochs`: Number of training epochs (default: 8)
- `--batch-size`: Batch size (default: 4)
- `--learning-rate`: Learning rate (default: 1e-5)
- `-o, --output`: Output directory name

### Using Your Trained Model

Interact with your trained model:

```bash
# Interactive chat mode
python ask.py my_model --interactive

# Ask a single question
python ask.py my_model --question "What is machine learning?"

# Process questions from file
python ask.py my_model --file questions.txt

# Adjust generation settings
python ask.py my_model --interactive --temperature 0.7
```

**Generation Options:**
- `--interactive`: Start interactive chat session
- `--question`: Ask a single question and exit
- `--file`: Process questions from a text file
- `--temperature`: Randomness in generation (0.1=focused, 1.0=creative)

## Data Formats

### Training Data
Plain text files for training. You can use:
- Single files: `example/raw/science_article_01.txt`
- Directories: `example/raw/` (trains on all .txt files)
- Your own text files

### Fine-tuning Data
JSON format with instruction-output pairs:
```json
[
    {
        "instruction": "What is the capital of France?",
        "output": "The capital of France is Paris."
    },
    {
        "instruction": "Explain photosynthesis.",
        "output": "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll..."
    }
]
```

## Development

### Running Tests
```bash
source venv/bin/activate
pytest
```

### Code Quality
```bash
# Format code
black .

# Type checking  
mypy src/

# Linting
flake8 src/
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for complete dependency list

## Troubleshooting

### Common Issues

**Training is slow**: Reduce model size with `--embedding-dim 256 --num-layers 4`

**Out of memory**: Reduce `--batch-size` or `--context-size`

**Poor results**: Try more training data, longer training (`--epochs`), or larger model

**Generation issues**: Adjust `--temperature` (lower=more focused, higher=more creative)
