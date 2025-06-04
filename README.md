# GPT Language Model

A production-quality PyTorch implementation of GPT (Generative Pre-trained Transformer) models with clean, simple interface.

## Quick Start

The project provides three main scripts for all your language model needs:

### 1. Train a Model
```bash
python train.py example/raw/example.txt -o my_model
```

### 2. Fine-tune on Q&A Data  
```bash
python fine-tune.py my_model example/fine-tuned/story_qa_dataset.json -o fine_tuned_model
```

### 3. Ask Questions
```bash
python ask.py fine_tuned_model --interactive
```

## Installation

```bash
git clone <repository-url>
cd language-model
pip install -r requirements.txt
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
# Basic training
python train.py example/raw/example.txt

# Custom model architecture
python train.py example/raw/example.txt \
    --vocab-size 8000 \
    --embedding-dim 512 \
    --num-heads 8 \
    --num-layers 8 \
    --iterations 5000 \
    --batch-size 16 \
    -o custom_model
```

**Options:**
- `--vocab-size`: Vocabulary size (default: 1500)
- `--embedding-dim`: Embedding dimension (default: 192)
- `--num-heads`: Number of attention heads (default: 6)
- `--num-layers`: Number of layers (default: 4)
- `--context-size`: Context window size (default: 96)
- `--iterations`: Training iterations (default: 300)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 3e-4)
- `-o, --output`: Output directory name

### Fine-tuning a Model

Fine-tune a trained model on Q&A data:

```bash
# Basic fine-tuning
python fine-tune.py my_model example/fine-tuned/story_qa_dataset.json

# Custom fine-tuning
python fine-tune.py my_model example/fine-tuned/story_qa_dataset.json \
    --epochs 10 \
    --batch-size 8 \
    --learning-rate 1e-5 \
    -o specialized_model
```

**Options:**
- `--epochs`: Number of training epochs (default: 8)
- `--batch-size`: Batch size (default: 4)
- `--learning-rate`: Learning rate (default: 1e-5)
- `-o, --output`: Output directory name

### Asking Questions

Interact with your fine-tuned model:

```bash
# Interactive mode
python ask.py fine_tuned_model --interactive

# Single question
python ask.py fine_tuned_model --question "What is the meaning of life?"

# Batch processing
python ask.py fine_tuned_model --file questions.txt

# Custom generation settings
python ask.py fine_tuned_model --interactive --temperature 0.7
```

**Options:**
- `--interactive`: Start interactive Q&A session
- `--question`: Ask a single question
- `--file`: Process questions from file
- `--temperature`: Generation randomness (default: 0.7)

## Data Formats

### Training Data
Plain text files for training:
```
example/raw/example.txt
```

### Fine-tuning Data
JSON format with question-answer pairs:
```json
[
    {
        "instruction": "What is the capital of France?",
        "output": "The capital of France is Paris."
    },
    {
        "instruction": "Explain photosynthesis.",
        "output": "Photosynthesis is the process by which plants convert sunlight into energy..."
    }
]
```

## Testing

Run the test suite:
```bash
pytest
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- See `requirements.txt` for full dependencies

## License

MIT License - see LICENSE file for details.
