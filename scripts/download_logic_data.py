#!/usr/bin/env python3
"""Download and format logic-focused datasets with chain-of-thought reasoning.

Combines:
- GSM8K: Grade school math with step-by-step solutions
- MATH: Competition mathematics
- LogiQA: Logical reasoning questions
- ARC-Challenge: Science reasoning
- Subset of general instruction data
"""

import json
import random
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "logic_cot"
INSTRUCT_DATA = PROJECT_ROOT / "data" / "instruct_1m" / "data.jsonl"


def format_gsm8k(example):
    """Format GSM8K with chain-of-thought."""
    question = example['question']
    # GSM8K has step-by-step answers ending with #### answer
    answer = example['answer']

    # Split into reasoning and final answer
    if '####' in answer:
        reasoning, final = answer.rsplit('####', 1)
        reasoning = reasoning.strip()
        final = final.strip()
    else:
        reasoning = answer
        final = ""

    return {
        'instruction': 'Solve this math problem step by step. Show your reasoning.',
        'input': question,
        'output': f"Let me work through this step by step:\n\n{reasoning}\n\nTherefore, the answer is: {final}"
    }


def format_math(example):
    """Format MATH dataset with chain-of-thought."""
    problem = example['problem']
    solution = example['solution']

    return {
        'instruction': 'Solve this math problem step by step. Show your reasoning.',
        'input': problem,
        'output': f"Let me work through this step by step:\n\n{solution}"
    }


def format_logiqa(example):
    """Format LogiQA for logical reasoning."""
    context = example.get('context', '')
    question = example.get('question', '')
    choices = example.get('options', [])
    answer_idx = example.get('label', 0)

    # Format choices
    choice_labels = ['A', 'B', 'C', 'D']
    choices_text = '\n'.join([f"{choice_labels[i]}. {c}" for i, c in enumerate(choices)])

    # Get correct answer
    correct_answer = choices[answer_idx] if answer_idx < len(choices) else choices[0]
    correct_label = choice_labels[answer_idx] if answer_idx < len(choice_labels) else 'A'

    input_text = f"{context}\n\nQuestion: {question}\n\nChoices:\n{choices_text}"

    return {
        'instruction': 'Analyze this logical reasoning problem step by step.',
        'input': input_text,
        'output': f"Let me analyze this step by step:\n\n1. First, I'll identify the key information from the context.\n2. Then, I'll evaluate each choice against the given information.\n3. Finally, I'll determine which answer logically follows.\n\nBased on my analysis, the answer is {correct_label}: {correct_answer}"
    }


def format_arc(example):
    """Format ARC-Challenge for science reasoning."""
    question = example.get('question', '')
    choices = example.get('choices', {})
    answer_key = example.get('answerKey', 'A')

    # Format choices
    labels = choices.get('label', [])
    texts = choices.get('text', [])
    choices_text = '\n'.join([f"{l}. {t}" for l, t in zip(labels, texts)])

    # Get correct answer
    try:
        answer_idx = labels.index(answer_key)
        correct_answer = texts[answer_idx]
    except (ValueError, IndexError):
        correct_answer = texts[0] if texts else ""

    input_text = f"{question}\n\nChoices:\n{choices_text}"

    return {
        'instruction': 'Answer this science question with reasoning.',
        'input': input_text,
        'output': f"Let me think through this:\n\nThe question asks about a scientific concept. Analyzing each option:\n\nThe correct answer is {answer_key}: {correct_answer}\n\nThis is because it aligns with the scientific principles involved."
    }


def load_instruct_subset(num_examples=200000):
    """Load a subset of general instruction data."""
    examples = []
    if INSTRUCT_DATA.exists():
        print(f"Loading instruction data from {INSTRUCT_DATA}...")
        with open(INSTRUCT_DATA, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()

        # Random sample
        if len(all_lines) > num_examples:
            sampled_lines = random.sample(all_lines, num_examples)
        else:
            sampled_lines = all_lines

        for line in sampled_lines:
            try:
                example = json.loads(line.strip())
                # Add step-by-step prefix to some examples
                if random.random() < 0.3:  # 30% get CoT formatting
                    example['instruction'] = example.get('instruction', '') + " Think step by step."
                examples.append(example)
            except json.JSONDecodeError:
                continue

    return examples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_examples = []

    # GSM8K - Grade school math
    print("Loading GSM8K...")
    try:
        gsm8k = load_dataset("openai/gsm8k", "main", split="train")
        gsm8k_examples = [format_gsm8k(ex) for ex in gsm8k]
        print(f"  GSM8K: {len(gsm8k_examples)} examples")
        all_examples.extend(gsm8k_examples)
    except Exception as e:
        print(f"  Failed to load GSM8K: {e}")

    # MATH - Competition math
    print("Loading MATH...")
    try:
        math_ds = load_dataset("lighteval/MATH", "all", split="train", trust_remote_code=True)
        math_examples = [format_math(ex) for ex in math_ds]
        print(f"  MATH: {len(math_examples)} examples")
        all_examples.extend(math_examples)
    except Exception as e:
        print(f"  Failed to load MATH: {e}")

    # LogiQA - Logical reasoning
    print("Loading LogiQA...")
    try:
        logiqa = load_dataset("lucasmccabe/logiqa", split="train")
        logiqa_examples = [format_logiqa(ex) for ex in logiqa]
        print(f"  LogiQA: {len(logiqa_examples)} examples")
        all_examples.extend(logiqa_examples)
    except Exception as e:
        print(f"  Failed to load LogiQA: {e}")

    # ARC-Challenge - Science reasoning
    print("Loading ARC-Challenge...")
    try:
        arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        arc_examples = [format_arc(ex) for ex in arc]
        print(f"  ARC-Challenge: {len(arc_examples)} examples")
        all_examples.extend(arc_examples)
    except Exception as e:
        print(f"  Failed to load ARC: {e}")

    # General instruction data subset
    print("Loading instruction data subset...")
    instruct_examples = load_instruct_subset(200000)
    print(f"  Instruction: {len(instruct_examples)} examples")
    all_examples.extend(instruct_examples)

    # Shuffle
    random.shuffle(all_examples)

    # Save
    output_file = OUTPUT_DIR / "data.jsonl"
    print(f"\nSaving {len(all_examples)} examples to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    # Stats
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {len(all_examples):,}")

    # Count by type
    cot_count = sum(1 for ex in all_examples if 'step by step' in ex.get('output', '').lower())
    print(f"  Chain-of-thought examples: {cot_count:,} ({100*cot_count/len(all_examples):.1f}%)")


if __name__ == "__main__":
    main()
