#!/usr/bin/env python3
"""
Generate a large fine-tuning dataset from raw text files.
This script processes various content types and creates instruction-following examples.
"""

import json
import os
import re
import random
from pathlib import Path
from typing import List, Dict, Any

def load_text_file(filepath: str) -> str:
    """Load text content from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return ""

def extract_qa_pairs(content: str) -> List[Dict[str, str]]:
    """Extract Q&A pairs from Q&A formatted text."""
    examples = []
    lines = content.split('\n')
    current_q = None
    
    # Alternative instruction templates
    instruction_templates = [
        "Answer the following question based on your knowledge.",
        "Provide a clear and accurate answer to this question.",
        "Please answer this question with the correct information.",
        "Give a helpful response to this question.",
        "Answer this question accurately and concisely."
    ]
    
    for line in lines:
        line = line.strip()
        if line.startswith('Q:'):
            current_q = line[2:].strip()
        elif line.startswith('A:') and current_q:
            answer = line[2:].strip()
            
            # Create multiple variations with different instructions
            for template in instruction_templates[:2]:  # Use 2 templates per Q&A
                examples.append({
                    "instruction": template,
                    "input": current_q,
                    "output": answer
                })
            
            # Add expanded answer version
            examples.append({
                "instruction": "Provide a detailed explanation for this question.",
                "input": current_q,
                "output": f"{answer} This is an important concept to understand as it relates to fundamental knowledge in this area."
            })
            
            current_q = None
    
    return examples

def create_instruction_examples(content: str, filename: str) -> List[Dict[str, str]]:
    """Convert instruction content into training examples."""
    examples = []
    
    # Multiple instruction templates
    instruction_templates = [
        "Provide step-by-step instructions for the following task.",
        "Explain how to do the following task step by step.",
        "Give detailed instructions for this task.",
        "Walk me through the process of doing this task.",
        "What are the steps to complete this task?"
    ]
    
    # Split by common instruction patterns
    sections = re.split(r'\n(?=\w+[:\n])', content)
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        lines = section.split('\n')
        if len(lines) < 2:
            continue
            
        # Extract title/topic and steps
        title = lines[0].rstrip(':')
        steps = []
        
        for line in lines[1:]:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Clean up numbering/bullets
                cleaned = re.sub(r'^[\d\-•\.\)\s]+', '', line).strip()
                if cleaned:
                    steps.append(cleaned)
        
        if steps:
            formatted_steps = '\n'.join(f"{i+1}. {step}" for i, step in enumerate(steps))
            
            # Create multiple variations
            for template in instruction_templates:
                examples.append({
                    "instruction": template,
                    "input": title,
                    "output": formatted_steps
                })
            
            # Add variations with different phrasings
            examples.extend([
                {
                    "instruction": "I need help with a task. Can you guide me?",
                    "input": f"How do I {title.lower()}?",
                    "output": formatted_steps
                },
                {
                    "instruction": "Provide a tutorial for this process.",
                    "input": f"Tutorial: {title}",
                    "output": formatted_steps
                },
                {
                    "instruction": "Break down this process into manageable steps.",
                    "input": title,
                    "output": formatted_steps
                }
            ])
    
    return examples

def generate_reading_comprehension(content: str, filename: str) -> List[Dict[str, str]]:
    """Generate reading comprehension questions from any text content."""
    examples = []
    
    # Split content into manageable chunks
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 50]
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
    
    if not paragraphs or not sentences:
        return examples
    
    topic_name = filename.replace('.txt', '').replace('_', ' ').title()
    
    # Reading comprehension question types
    comprehension_types = [
        {
            "instruction": "Read this text and answer the question about it.",
            "input": f"Based on the text about {topic_name}, what are the main ideas?",
            "output": paragraphs[0] if paragraphs else "The main ideas include key concepts and their applications."
        },
        {
            "instruction": "Analyze this text and provide insights.",
            "input": f"What insights can you provide about {topic_name}?",
            "output": paragraphs[1] if len(paragraphs) > 1 else paragraphs[0]
        },
        {
            "instruction": "Explain the content of this text.",
            "input": f"Can you explain what this text about {topic_name} covers?",
            "output": f"This text covers {topic_name}, including: " + (paragraphs[0][:200] + "..." if len(paragraphs[0]) > 200 else paragraphs[0])
        },
        {
            "instruction": "Summarize the key points from this reading.",
            "input": f"What are the key points about {topic_name}?",
            "output": paragraphs[0] if paragraphs else "The key points include important concepts and their practical applications."
        }
    ]
    
    # Add specific detail questions
    if len(sentences) >= 3:
        detail_questions = [
            {
                "instruction": "Answer this specific question about the text.",
                "input": f"What specific details are mentioned about {topic_name}?",
                "output": random.choice(sentences[:5])
            },
            {
                "instruction": "Provide detailed information from this text.",
                "input": f"Give me detailed information about {topic_name}.",
                "output": '. '.join(sentences[:3]) + '.'
            },
            {
                "instruction": "Extract important information from this text.",
                "input": f"What important information is provided about {topic_name}?",
                "output": random.choice(sentences[:4])
            }
        ]
        comprehension_types.extend(detail_questions)
    
    examples.extend(comprehension_types)
    return examples

def create_explanation_examples(content: str, filename: str) -> List[Dict[str, str]]:
    """Create explanation and reasoning examples."""
    examples = []
    
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 30]
    if not paragraphs:
        return examples
    
    topic_name = filename.replace('.txt', '').replace('_', ' ').title()
    
    explanation_templates = [
        {
            "instruction": "Explain this concept clearly and thoroughly.",
            "input": f"Explain {topic_name} to me.",
            "output": paragraphs[0]
        },
        {
            "instruction": "Provide a detailed explanation of this topic.",
            "input": f"I want to understand {topic_name}. Can you explain it?",
            "output": paragraphs[0]
        },
        {
            "instruction": "Break down this concept for better understanding.",
            "input": f"Help me understand {topic_name}.",
            "output": paragraphs[0]
        },
        {
            "instruction": "Describe this topic in an informative way.",
            "input": f"Describe {topic_name} for me.",
            "output": paragraphs[0]
        }
    ]
    
    if len(paragraphs) > 1:
        explanation_templates.extend([
            {
                "instruction": "Provide comprehensive information about this topic.",
                "input": f"Tell me everything important about {topic_name}.",
                "output": ' '.join(paragraphs[:2])
            },
            {
                "instruction": "Give a thorough explanation of this subject.",
                "input": f"I need a thorough explanation of {topic_name}.",
                "output": ' '.join(paragraphs[:2])
            }
        ])
    
    examples.extend(explanation_templates)
    return examples

def create_conversation_examples(content: str) -> List[Dict[str, str]]:
    """Convert conversations into dialogue training examples."""
    examples = []
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    # Multiple instruction variations for conversations
    conversation_instructions = [
        "Respond naturally and helpfully to this conversational message.",
        "Provide a friendly and appropriate response to this message.",
        "Reply to this in a conversational and helpful manner.",
        "Give a natural response to this conversational prompt.",
        "Respond appropriately to this dialogue."
    ]
    
    # Group lines into conversation pairs
    for i in range(0, len(lines) - 1, 2):
        if i + 1 < len(lines):
            user_msg = lines[i].strip('"')
            assistant_msg = lines[i + 1].strip('"')
            
            # Create multiple variations of each conversation
            for instruction in conversation_instructions[:3]:  # Use 3 variations per conversation
                examples.append({
                    "instruction": instruction,
                    "input": user_msg,
                    "output": assistant_msg
                })
            
            # Add context-aware variations
            examples.extend([
                {
                    "instruction": "Continue this conversation in a natural way.",
                    "input": f"In a conversation, someone says: '{user_msg}' How would you respond?",
                    "output": assistant_msg
                },
                {
                    "instruction": "Provide a helpful conversational response.",
                    "input": f"Someone tells you: '{user_msg}' What's a good response?",
                    "output": assistant_msg
                }
            ])
    
    return examples

def generate_article_questions(content: str, topic_type: str) -> List[Dict[str, str]]:
    """Generate comprehensive questions from article content."""
    examples = []
    
    # Split into paragraphs and sentences
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    all_sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in all_sentences if s.strip() and len(s.strip()) > 20]
    
    # Expanded question templates
    templates = {
        "science": [
            "What is {} and how does it work?",
            "Explain the concept of {} in simple terms.",
            "What are the key principles behind {}?",
            "How does {} impact our daily lives?",
            "What are the main applications of {}?",
            "What are the benefits of studying {}?",
            "How is {} related to other scientific fields?",
            "What recent discoveries have been made about {}?",
            "What research methods are used to study {}?",
            "What are the limitations of current {} research?",
            "How might {} develop in the next decade?",
            "What ethical considerations surround {}?",
            "What role does {} play in modern medicine?",
            "How has our understanding of {} changed over time?",
            "What are the practical implications of {} research?"
        ],
        "history": [
            "What happened during {}?",
            "Who were the key figures in {}?",
            "What were the causes and effects of {}?",
            "How did {} shape the modern world?",
            "What lessons can we learn from {}?",
            "What was the significance of {}?",
            "How did {} influence later events?",
            "What were the social impacts of {}?",
            "What economic factors contributed to {}?",
            "How did {} affect different social classes?",
            "What role did technology play in {}?",
            "How is {} remembered today?",
            "What sources do we have about {}?",
            "How did {} compare to similar events?",
            "What were the long-term consequences of {}?"
        ],
        "technology": [
            "How does {} technology work?",
            "What are the benefits and risks of {}?",
            "How has {} evolved over time?",
            "What is the future of {}?",
            "How is {} changing our society?",
            "What industries use {} technology?",
            "What are the technical challenges of {}?",
            "How does {} compare to alternative technologies?",
            "What skills are needed to work with {}?",
            "How does {} affect privacy and security?",
            "What are the environmental impacts of {}?",
            "How accessible is {} technology?",
            "What regulations govern {} development?",
            "How does {} integrate with existing systems?",
            "What are the costs associated with {}?"
        ]
    }
    
    # Extract key concepts
    if paragraphs and topic_type in templates and sentences:
        first_para = paragraphs[0]
        
        # Enhanced keyword extraction
        keywords = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', first_para)
        keywords.extend(re.findall(r'\b(?:process|system|method|technique|theory|principle|research|study|analysis|development|innovation|discovery)\b', first_para, re.IGNORECASE))
        
        # Add topic-specific keywords
        if topic_type == "science":
            keywords.extend(re.findall(r'\b(?:experiment|hypothesis|data|evidence|research|laboratory|scientist|molecule|cell|DNA|protein|enzyme|reaction)\b', content, re.IGNORECASE))
        elif topic_type == "history":
            keywords.extend(re.findall(r'\b(?:civilization|empire|war|revolution|culture|society|government|leader|dynasty|period|era|century)\b', content, re.IGNORECASE))
        elif topic_type == "technology":
            keywords.extend(re.findall(r'\b(?:software|hardware|algorithm|computer|digital|artificial|intelligence|automation|robot|innovation|device)\b', content, re.IGNORECASE))
        
        # Remove duplicates and short keywords
        keywords = list(set([k for k in keywords if len(k) > 3]))
        
        if keywords:
            # Create 10-15 questions per article (increased from 5-8)
            num_questions = min(15, len(templates[topic_type]))
            selected_templates = random.sample(templates[topic_type], num_questions)
            
            for template in selected_templates:
                keyword = random.choice(keywords)
                question = template.format(keyword.lower())
                
                # Create more varied answers
                if len(sentences) >= 3:
                    # Use different sentence combinations
                    answer_sentences = random.sample(sentences[:8], min(4, len(sentences[:8])))
                    answer = '. '.join(answer_sentences) + '.'
                else:
                    answer = '. '.join(sentences) + '.'
                
                examples.append({
                    "instruction": f"Answer this {topic_type} question based on your knowledge.",
                    "input": question,
                    "output": answer
                })
        
        # Add multiple comprehension question types
        if len(paragraphs) >= 1:
            comprehension_examples = [
                {
                    "instruction": f"Summarize the main points of this {topic_type} text.",
                    "input": f"Summarize the key information about {topic_type}.",
                    "output": paragraphs[0] if paragraphs[0] else "This text discusses important concepts and findings in the field."
                },
                {
                    "instruction": f"What are the key takeaways from this {topic_type} information?",
                    "input": f"What should I know about {topic_type}?",
                    "output": paragraphs[0] if paragraphs[0] else "The key points include fundamental concepts and their applications."
                },
                {
                    "instruction": "Provide a comprehensive explanation of this topic.",
                    "input": f"Can you explain {topic_type} in detail?",
                    "output": ' '.join(paragraphs[:2]) if len(paragraphs) >= 2 else paragraphs[0]
                }
            ]
            
            if len(paragraphs) > 1:
                comprehension_examples.extend([
                    {
                        "instruction": f"Explain the importance of this {topic_type} topic.",
                        "input": f"Why is this {topic_type} topic important to understand?",
                        "output": paragraphs[1] if len(paragraphs) > 1 else "This topic is important because it contributes to our understanding of the world around us."
                    },
                    {
                        "instruction": "Compare and contrast the different aspects mentioned.",
                        "input": f"What are the different aspects of {topic_type}?",
                        "output": paragraphs[1] if len(paragraphs) > 1 else "There are various important aspects to consider."
                    },
                    {
                        "instruction": "Discuss the practical applications mentioned.",
                        "input": f"How is {topic_type} applied in practice?",
                        "output": paragraphs[2] if len(paragraphs) > 2 else paragraphs[1]
                    }
                ])
            
            examples.extend(comprehension_examples)
    
    return examples

def create_creative_prompts(content: str, content_type: str) -> List[Dict[str, str]]:
    """Create creative writing prompts from fiction/poetry content."""
    examples = []
    
    if content_type == "fiction":
        # Extract themes, characters, settings
        lines = content.split('\n')
        first_lines = [line.strip() for line in lines[:5] if line.strip()]
        
        if first_lines:
            # Create writing prompts based on the story
            examples.extend([
                {
                    "instruction": "Write a short creative story based on this prompt.",
                    "input": "Write a story that begins with: " + first_lines[0],
                    "output": content[:500] + "..." if len(content) > 500 else content
                },
                {
                    "instruction": "Continue this story in an engaging way.",
                    "input": first_lines[0] if first_lines else "Once upon a time...",
                    "output": '\n'.join(first_lines[1:3]) if len(first_lines) > 1 else "The story continued with unexpected twists and turns..."
                }
            ])
    
    elif content_type == "poetry":
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if lines:
            examples.append({
                "instruction": "Write a poem in response to this prompt or theme.",
                "input": "Write a poem similar in style to: " + lines[0],
                "output": content
            })
    
    return examples

def main():
    """Main function to generate the dataset."""
    raw_data_dir = Path("/Users/thomasmickley-doyle/Repos/language-model/example/raw")
    output_file = Path("/Users/thomasmickley-doyle/Repos/language-model/example/fine-tuned/instruction_dataset.json")
    
    all_examples = []
    
    # Process all files in the raw data directory
    for filepath in raw_data_dir.glob("*.txt"):
        print(f"Processing {filepath.name}...")
        content = load_text_file(str(filepath))
        
        if not content:
            continue
        
        filename = filepath.name
        
        # Process based on file type
        if filename.startswith("qa_examples"):
            examples = extract_qa_pairs(content)
            all_examples.extend(examples)
            
        elif filename.startswith("instructions"):
            examples = create_instruction_examples(content, filename)
            all_examples.extend(examples)
            
        elif filename.startswith("daily_conversation"):
            examples = create_conversation_examples(content)
            all_examples.extend(examples)
            
        elif filename.startswith("science_article") or any(x in filename for x in ["science", "biology", "physics", "chemistry", "medical", "quantum", "molecular", "climate", "environmental", "marine", "mathematics", "advanced_materials", "renewable_energy"]):
            examples = generate_article_questions(content, "science")
            all_examples.extend(examples)
            
        elif filename.startswith("history_article") or any(x in filename for x in ["history", "civilization", "mesopotamian", "renaissance", "silk_road", "library_alexandria", "notre_dame"]):
            examples = generate_article_questions(content, "history")
            all_examples.extend(examples)
            
        elif filename.startswith("technology_article") or any(x in filename for x in ["technology", "robotics", "ai_creative", "cryptocurrency", "digital_revolution", "cognitive_science"]):
            examples = generate_article_questions(content, "technology")
            all_examples.extend(examples)
            
        elif filename.startswith("fiction_story") or any(x in filename for x in ["fiction", "story", "cybercrime_thriller", "extended_story", "storytelling"]):
            examples = create_creative_prompts(content, "fiction")
            all_examples.extend(examples)
            
        elif filename.startswith("poetry"):
            examples = create_creative_prompts(content, "poetry")
            all_examples.extend(examples)
            
        # For ALL files, add reading comprehension and explanation examples
        reading_examples = generate_reading_comprehension(content, filename)
        all_examples.extend(reading_examples)
        
        explanation_examples = create_explanation_examples(content, filename)
        all_examples.extend(explanation_examples)
        
        # Catch any remaining files as general knowledge
        if not any(pattern in filename for pattern in ["qa_examples", "instructions", "daily_conversation", "science", "history", "technology", "fiction", "poetry"]):
            # Create general knowledge examples from remaining content
            examples = []
            if len(content) > 100:  # Only process substantial content
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                if paragraphs:
                    examples.append({
                        "instruction": "Provide information about this topic.",
                        "input": f"Tell me about {filename.replace('.txt', '').replace('_', ' ')}",
                        "output": paragraphs[0]
                    })
                    if len(paragraphs) > 1:
                        examples.append({
                            "instruction": "Explain this concept in detail.",
                            "input": f"Can you explain more about {filename.replace('.txt', '').replace('_', ' ')}?",
                            "output": paragraphs[1]
                        })
            all_examples.extend(examples)
    
    # Shuffle the examples for better training distribution
    random.shuffle(all_examples)
    
    # Save the dataset
    print(f"Generated {len(all_examples)} training examples")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved to {output_file}")
    print(f"Total examples: {len(all_examples)}")

if __name__ == "__main__":
    main()