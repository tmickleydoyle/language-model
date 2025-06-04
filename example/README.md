# Example Data Directory

This directory contains example data for training and fine-tuning the GPT language model, organized into two phases:

## Directory Structure

### `raw/`
Contains raw text data for initial language model training:
- **`example.txt`** - Expanded story about Beau the puppy in New Orleans (~2000+ words)
  - Used for pre-training the model on natural language patterns
  - Contains rich narrative with characters, settings, and dialogue
  - Training command: `python scripts/train.py --data-path example/raw/example.txt`

### `fine-tuned/`
Contains structured datasets for instruction fine-tuning:
- **`story_qa_dataset.json`** - Question-answer pairs in Alpaca format (40 Q&A pairs)
  - Used for fine-tuning the model to answer questions about the story
  - Format: `{"instruction": "...", "input": "question", "output": "answer"}`
  - Enables the model to demonstrate reading comprehension and specific knowledge

## Training Workflow

### Phase 1: Pre-training on Raw Text
```bash
# Train on the expanded story text
python scripts/train.py --data-path example/raw/example.txt

# Optimal parameters for this dataset:
python scripts/train.py \
    --data-path example/raw/example.txt \
    --n-embd 64 \
    --n-head 2 \
    --n-layer 2 \
    --vocab-size 256 \
    --batch-size 32 \
    --learning-rate 3e-4
```

### Phase 2: Fine-tuning on Q&A Data
```bash
# Future implementation: Fine-tune on question-answer pairs
# python fine_tune.py --data-path example/fine-tuned/story_qa_dataset.json --base-model checkpoints/best.pt
```

## Data Details

### Raw Story Content
- **Characters**: Beau (puppy), Madame Delphine (owner), Pierre (friend), Sister Marie-Claire, Louis, children
- **Setting**: New Orleans - Royal Street cottage, Ursuline Convent, French Market, Garden District
- **Timeline**: Spans seasons and years showing character development
- **Themes**: Friendship, community, cultural heritage, growth

### Q&A Dataset
- **Coverage**: Character details, plot events, settings, relationships, timeline
- **Question Types**: Who/what/where/when/why/how questions
- **Format**: Alpaca instruction format for supervised fine-tuning
- **Purpose**: Teaching the model to answer specific questions about story content

## Usage Notes

- Use `raw/` data for initial model training to learn language patterns
- Use `fine-tuned/` data for specialized task training (Q&A, instruction following)
- The story content provides rich context for testing text generation quality
- The Q&A pairs enable evaluation of reading comprehension capabilities
