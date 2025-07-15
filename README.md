# GPT Language Model

A PyTorch implementation of GPT (Generative Pre-trained Transformer) models with modern architecture improvements, streaming data support, and a clean, simple interface.

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

**Local File Training:**
```bash
# Train on a single file
python train.py example/raw/science_article_01.txt -o my_model

# Train on all files in directory  
python train.py example/raw/ -o comprehensive_model
```

**Streaming Data Training (Recommended for Large Models):**
```bash
# Train on massive datasets without local storage
python train_streaming.py --dataset-config large_mixed \
    --embedding-dim 768 --num-heads 12 --num-layers 12 \
    --context-size 2048 --batch-size 8 --epochs 20 \
    --cache-size 50000 -o large_model

# Quick Wikipedia training
python train_streaming.py --dataset-config wikipedia_only \
    --embedding-dim 256 --num-heads 8 --num-layers 6 \
    -o wiki_model
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

## Streaming Data System

### Available Data Sources
- **OpenWebText** - 40GB web content (GPT-2 training data)
- **The Pile** - 800GB diverse text (academic, books, news, code)
- **C4** - 750GB cleaned web text (T5 training data)
- **Wikipedia** - Live articles via API
- **Project Gutenberg** - Classic books via API

### Dataset Configurations
```bash
--dataset-config small_mixed    # 3K samples (Wikipedia + OpenWebText)
--dataset-config medium_mixed   # 20K samples (3 sources)
--dataset-config large_mixed    # 100K samples (4 sources)
--dataset-config wikipedia_only # 20K Wikipedia articles
--dataset-config web_focused    # 50K web content samples
```

### Benefits
✅ **No local storage** - Stream directly from APIs  
✅ **Massive datasets** - Access to petabytes of text  
✅ **Always fresh** - New content every training run  
✅ **Memory efficient** - Configurable cache sizes  
✅ **Multiple sources** - Mix different data types

### Test Streaming
```bash
python test_streaming.py        # Test all streaming components
python demo_streaming.py        # Quick Wikipedia demo
python quick_streaming_demo.py  # Full training demo
```

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
│   │   ├── dataset.py       # Local file datasets
│   │   ├── streaming_data_loader.py  # API data streaming
│   │   └── streaming_dataset.py      # Streaming PyTorch datasets
│   ├── training/            # Training infrastructure
│   └── utils/               # Utility functions
├── tests/                   # Test suite
├── example/                 # Training datasets
│   ├── raw/                 # Raw text for training
│   └── fine-tuned/          # Q&A datasets
├── train.py                 # Local file training script
├── train_streaming.py       # Streaming data training script
├── fine-tune.py             # Fine-tuning script
├── ask.py                   # Interactive Q&A script
├── test_streaming.py        # Test streaming components
├── demo_streaming.py        # Quick streaming demo
├── quick_streaming_demo.py  # Full streaming training demo
└── README.md                # This file
```

## Usage Guide

### Training a Model

#### Local File Training
Train a GPT model from scratch using local files:

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

#### Streaming Data Training
Train on massive datasets without local storage:

```bash
# Large-scale training with streaming data
python train_streaming.py --dataset-config large_mixed \
    --embedding-dim 768 \
    --num-heads 12 \
    --num-layers 12 \
    --context-size 2048 \
    --batch-size 8 \
    --epochs 20 \
    --cache-size 50000 \
    -o large_streaming_model

# Wikipedia-only training
python train_streaming.py --dataset-config wikipedia_only \
    --embedding-dim 256 \
    --num-heads 8 \
    --num-layers 6 \
    --refresh-rate 3 \
    -o wiki_model

# Quick test with small model
python train_streaming.py --dataset-config small_mixed \
    --embedding-dim 128 \
    --num-heads 4 \
    --num-layers 3 \
    --cache-size 1000 \
    -o test_model
```

**Local Training Options:**
- `--embedding-dim`: Model dimension (default: 128)
- `--num-heads`: Number of attention heads (default: 4)  
- `--num-layers`: Number of transformer layers (default: 3)
- `--context-size`: Context window size (default: 128)
- `--epochs`: Training epochs (default: 50)
- `--batch-size`: Batch size (default: 16)
- `--learning-rate`: Learning rate (default: 5e-4)
- `--dropout`: Dropout rate (default: 0.3)
- `-o, --output`: Output directory name

**Streaming Training Options:**
- `--dataset-config`: Data source configuration (small_mixed, large_mixed, etc.)
- `--cache-size`: Number of samples to cache in memory (default: 10000)
- `--refresh-rate`: Refresh cache every N epochs (default: 5)
- `--use-cache`: Use cached streaming dataset (default: True)
- All local training options also apply

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
- **For streaming data:**
  - datasets>=2.0.0
  - wikipedia>=1.4.0  
  - requests>=2.25.0
- See `requirements.txt` for complete dependency list

## Troubleshooting

### Common Issues

**Training is slow**: Reduce model size with `--embedding-dim 256 --num-layers 4`

**Out of memory**: Reduce `--batch-size` or `--context-size`

**Poor results**: Try more training data, longer training (`--epochs`), or larger model

**Generation issues**: Adjust `--temperature` (lower=more focused, higher=more creative)
