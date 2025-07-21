# RLHF Data Organization

This directory contains all data and utilities for Reinforcement Learning from Human Feedback (RLHF) training.

## 📁 Directory Structure

```
data/rlhf/
├── README.md                    # This file
├── datasets/                    # All preference datasets
│   ├── curated/                # Hand-crafted, high-quality examples
│   │   ├── preference_dataset_starter.json      # 8 comprehensive examples
│   │   ├── preferences_instruction_following.json # 5 instruction examples
│   │   ├── preferences_safety.json              # 2 safety examples
│   │   └── preferences_reasoning.json           # 1 reasoning example
│   ├── synthetic/              # Auto-generated examples
│   │   └── large_synthetic_preferences.json     # 100 synthetic examples
│   ├── combined/               # Merged datasets
│   │   └── combined_preference_dataset.json     # 108 total examples
│   └── user_collected/         # Your custom preferences (created when used)
├── prompts/                    # Prompt collections for data generation
│   └── collection_prompts.txt  # 25 diverse prompts for preference collection
└── examples/                   # Scripts for generating data
    ├── create_preference_dataset.py      # Creates curated examples
    └── generate_large_synthetic_dataset.py # Creates synthetic examples
```

## 📊 Dataset Overview

### Curated Datasets (High Quality)
- **preference_dataset_starter.json** (8 examples): Comprehensive starter set
  - Instruction following: 5 examples
  - Safety/harmlessness: 2 examples  
  - Reasoning: 1 example
- **Category-specific datasets**: Split by topic for targeted training

### Synthetic Datasets (Large Scale)
- **large_synthetic_preferences.json** (100 examples): Auto-generated examples
  - Detailed vs Brief: 26 examples
  - Helpful vs Vague: 31 examples
  - Accurate vs Incorrect: 24 examples
  - Structured vs Rambling: 19 examples

### Combined Datasets (Best of Both)
- **combined_preference_dataset.json** (108 examples): Curated + Synthetic

## 🚀 Usage Examples

### Train with curated data (recommended start)
```bash
python train_rlhf.py --method dpo \
                     --model_path your_model.pth \
                     --preference_data data/rlhf/datasets/curated/preference_dataset_starter.json
```

### Train with large dataset  
```bash
python train_rlhf.py --method dpo \
                     --model_path your_model.pth \
                     --preference_data data/rlhf/datasets/combined/combined_preference_dataset.json
```

### Collect your own preferences
```bash
python train_rlhf.py --method dpo \
                     --model_path your_model.pth \
                     --collect_preferences \
                     --prompts_file data/rlhf/prompts/collection_prompts.txt
```

### Generate more synthetic data
```bash
cd data/rlhf/examples
python generate_large_synthetic_dataset.py
```

## 📋 Data Format

Each preference dataset contains JSON objects with this structure:
```json
{
  "prompt": "Question or instruction text: ",
  "chosen": "High-quality response that should be preferred",
  "rejected": "Lower-quality response that should be avoided", 
  "metadata": {
    "category": "instruction_following|safety|reasoning",
    "quality": "helpful_vs_vague|detailed_vs_brief|etc",
    "source": "curated|synthetic|user_collected"
  }
}
```

## 🎯 Recommended Training Pipeline

1. **Start Small**: Train with `preference_dataset_starter.json` (8 examples)
2. **Scale Up**: Use `combined_preference_dataset.json` (108 examples)  
3. **Customize**: Collect domain-specific preferences with `collection_prompts.txt`
4. **Iterate**: Generate more synthetic data as needed

## 🔄 Data Generation

- **Curated Examples**: Run `examples/create_preference_dataset.py`
- **Synthetic Examples**: Run `examples/generate_large_synthetic_dataset.py` 
- **Interactive Collection**: Use `--collect_preferences` flag in training script

## 📈 Quality Guidelines

**Good Chosen Responses:**
- Helpful and informative
- Well-structured with clear explanations
- Appropriate length (not too brief, not excessive)
- Safe and ethical
- Accurate information

**Good Rejected Responses:**
- Too brief or vague
- Off-topic or unhelpful
- Repetitive or rambling
- Potentially harmful
- Factually incorrect

## 🛠️ Extending the Dataset

To add more preference examples:

1. **Add prompts** to `prompts/collection_prompts.txt`
2. **Run data generation scripts** in `examples/`
3. **Manually create** high-quality examples in `datasets/curated/`
4. **Use interactive collection** to gather domain-specific preferences

## 📝 Notes

- All datasets are in JSON format for easy parsing
- Metadata helps track data sources and quality types
- Directory structure scales for large datasets
- Scripts are modular and reusable