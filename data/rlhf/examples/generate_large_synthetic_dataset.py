#!/usr/bin/env python3
"""
Generate a large synthetic preference dataset for RLHF training.

This script creates hundreds of synthetic preference examples using
various heuristics and quality measures.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.preference_dataset import PreferenceExample

def create_diverse_prompts() -> List[str]:
    """Create a large, diverse set of prompts for synthetic generation."""
    
    # Educational prompts
    educational = [
        "Explain how photosynthesis works: ",
        "What is the theory of relativity? ",
        "How does the immune system work? ",
        "Explain the water cycle: ",
        "What causes earthquakes? ",
        "How do computers process information? ",
        "What is DNA and why is it important? ",
        "Explain how vaccines work: ",
        "What is climate change? ",
        "How does the human brain work? ",
        "What is artificial intelligence? ",
        "Explain the carbon cycle: ",
        "How do batteries work? ",
        "What is quantum physics? ",
        "How does evolution work? ",
        "What is the Internet? ",
        "Explain photosynthesis to a child: ",
        "How do airplanes fly? ",
        "What is renewable energy? ",
        "How does GPS work? "
    ]
    
    # Creative writing prompts
    creative = [
        "Write a story about a time traveler: ",
        "Describe a perfect day: ",
        "Write about a world where colors have sounds: ",
        "Tell a story about finding a mysterious object: ",
        "Describe your dream vacation: ",
        "Write about a conversation with your future self: ",
        "Create a story about a magical library: ",
        "Describe a city floating in the clouds: ",
        "Write about an unlikely friendship: ",
        "Tell a story about discovering a hidden talent: ",
        "Describe a world without gravity: ",
        "Write about a day when animals could talk: ",
        "Create a story about a robot learning emotions: ",
        "Describe the last bookstore on Earth: ",
        "Write about a person who can see memories: ",
        "Tell a story about living underwater: ",
        "Describe a museum of lost things: ",
        "Write about a character who ages backwards: ",
        "Create a story about musical weather: ",
        "Describe a world where dreams are visible: "
    ]
    
    # Practical advice prompts
    practical = [
        "How can I improve my sleep quality? ",
        "What's the best way to learn a new language? ",
        "How do I prepare for a job interview? ",
        "What are good study techniques? ",
        "How can I be more productive? ",
        "What's the best way to save money? ",
        "How do I start exercising regularly? ",
        "What are good cooking tips for beginners? ",
        "How can I improve my public speaking? ",
        "What's the best way to network professionally? ",
        "How do I manage stress effectively? ",
        "What are good time management strategies? ",
        "How can I improve my writing skills? ",
        "What's the best way to stay motivated? ",
        "How do I build healthy relationships? ",
        "What are good habits to develop? ",
        "How can I improve my memory? ",
        "What's the best way to learn new skills? ",
        "How do I overcome procrastination? ",
        "What are effective communication tips? "
    ]
    
    # Problem-solving prompts
    problem_solving = [
        "How would you solve world hunger? ",
        "What's the best approach to climate change? ",
        "How can we reduce plastic pollution? ",
        "What's the solution to traffic congestion? ",
        "How can we improve education systems? ",
        "What's the best way to reduce inequality? ",
        "How can we make cities more sustainable? ",
        "What's the solution to the housing crisis? ",
        "How can we improve healthcare access? ",
        "What's the best way to promote renewable energy? ",
        "How can we reduce food waste? ",
        "What's the solution to unemployment? ",
        "How can we improve mental health support? ",
        "What's the best way to combat misinformation? ",
        "How can we protect endangered species? ",
        "What's the solution to water scarcity? ",
        "How can we improve elder care? ",
        "What's the best way to support small businesses? ",
        "How can we reduce digital addiction? ",
        "What's the solution to cybersecurity threats? "
    ]
    
    # Technical prompts
    technical = [
        "Explain how machine learning works: ",
        "What is cloud computing? ",
        "How does blockchain technology work? ",
        "What is the difference between AI and ML? ",
        "Explain how databases work: ",
        "What is cybersecurity? ",
        "How do search engines work? ",
        "What is software engineering? ",
        "Explain how the Internet works: ",
        "What is data science? ",
        "How do programming languages work? ",
        "What is user experience design? ",
        "Explain how apps are developed: ",
        "What is network security? ",
        "How do operating systems work? ",
        "What is version control? ",
        "Explain how APIs work: ",
        "What is agile development? ",
        "How do websites work? ",
        "What is artificial neural networks? "
    ]
    
    return educational + creative + practical + problem_solving + technical

def generate_response_variations(prompt: str, base_response: str) -> List[str]:
    """Generate variations of a response with different quality levels."""
    
    # High quality version (chosen)
    high_quality = enhance_response(base_response)
    
    # Generate lower quality versions (rejected)
    variations = []
    
    # Too short version
    words = base_response.split()
    if len(words) > 10:
        short_version = " ".join(words[:len(words)//3]) + "."
        variations.append(short_version)
    
    # Repetitive version
    repetitive = base_response + " " + base_response.split('.')[0] + ". " + base_response.split('.')[0] + "."
    variations.append(repetitive)
    
    # Vague version
    vague_words = ["something", "things", "stuff", "maybe", "probably", "somehow"]
    vague_response = f"Well, {random.choice(vague_words)} related to this involves {random.choice(vague_words)} that {random.choice(vague_words)} works with other {random.choice(vague_words)}."
    variations.append(vague_response)
    
    # Off-topic version
    off_topic_responses = [
        "That reminds me of a completely different topic. Let me tell you about cats instead.",
        "I don't really know about that, but did you know that bananas are berries?",
        "That's interesting, but I prefer to talk about the weather today.",
        "Speaking of that, have you ever wondered why the sky is blue?",
        "That's a good question, but I think you should ask someone else."
    ]
    variations.append(random.choice(off_topic_responses))
    
    return [high_quality] + variations

def enhance_response(response: str) -> str:
    """Enhance a response to be more helpful and detailed."""
    
    # Add structure if missing
    if ":" not in response and len(response) > 50:
        sentences = response.split('. ')
        if len(sentences) > 2:
            # Add a summary or introduction
            enhanced = f"Here's a clear explanation:\n\n{response}"
            
            # Add examples if appropriate
            if any(word in response.lower() for word in ["how", "what", "explain"]):
                enhanced += "\n\nFor example: This concept applies in many real-world situations."
                
            return enhanced
    
    return response

def create_base_responses() -> Dict[str, str]:
    """Create base responses for different types of prompts."""
    
    responses = {
        # Educational responses
        "Explain how photosynthesis works: ": "Photosynthesis is the process by which plants convert sunlight into energy. Plants use chlorophyll to absorb light energy, combine carbon dioxide from air with water from soil, and produce glucose (sugar) for energy plus oxygen as a byproduct. This process occurs mainly in leaves and is essential for life on Earth since it produces the oxygen we breathe.",
        
        "What is the theory of relativity? ": "Einstein's theory of relativity consists of two parts: special and general relativity. Special relativity shows that space and time are connected as spacetime, and nothing can travel faster than light. General relativity describes gravity not as a force but as curvature in spacetime caused by mass and energy. This theory revolutionized physics and led to insights about black holes, GPS technology, and the expansion of the universe.",
        
        "How does the immune system work? ": "The immune system protects your body from harmful invaders like bacteria, viruses, and toxins. It has two main parts: innate immunity (immediate response like white blood cells) and adaptive immunity (learned response like antibodies). When pathogens enter your body, immune cells identify and attack them, while memory cells remember the invader for faster response next time. This is how vaccines work - they train your immune system safely.",
        
        # Creative responses
        "Write a story about a time traveler: ": "Sarah discovered the old pocket watch in her grandmother's attic, its silver surface etched with strange symbols. When she wound it, the world around her began to shimmer and fade. Suddenly, she stood in the same attic, but everything looked different - newer, brighter. Through the window, she saw horse-drawn carriages instead of cars. She had traveled back to 1920, and her grandmother, now a young woman, was walking up the stairs.",
        
        "Describe a perfect day: ": "A perfect day begins with gentle sunlight streaming through the windows and the aroma of fresh coffee. The morning starts slowly with a good book, followed by a walk in nature where birds sing and flowers bloom. The afternoon brings time with loved ones, sharing laughter and meaningful conversations. Evening arrives with a beautiful sunset, a delicious home-cooked meal, and the satisfaction of having connected with both the world and the people who matter most.",
        
        # Practical responses
        "How can I improve my sleep quality? ": "To improve sleep quality, establish a consistent bedtime routine and sleep schedule. Create a cool, dark, quiet bedroom environment. Avoid screens, caffeine, and large meals before bedtime. Regular exercise helps, but not too close to sleep time. Try relaxation techniques like deep breathing or gentle stretching. Keep a sleep diary to identify patterns. If problems persist, consider consulting a healthcare provider as sleep disorders may need professional treatment.",
        
        "What's the best way to learn a new language? ": "Effective language learning combines multiple approaches: start with basic vocabulary and common phrases, practice speaking from day one (even if just to yourself), use language learning apps for daily practice, consume media in the target language (movies, music, podcasts), and find conversation partners online or locally. Set realistic goals, be consistent, and don't fear making mistakes - they're part of learning. Immersion experiences, even virtual ones, accelerate progress significantly.",
        
        # Technical responses
        "Explain how machine learning works: ": "Machine learning enables computers to learn patterns from data without explicit programming. The process involves feeding algorithms large amounts of data so they can identify patterns and make predictions. There are three main types: supervised learning (learning from labeled examples), unsupervised learning (finding hidden patterns), and reinforcement learning (learning through trial and error). The algorithm improves its performance over time as it processes more data, similar to how humans learn from experience."
    }
    
    return responses

def generate_synthetic_preferences(prompts: List[str], base_responses: Dict[str, str]) -> List[PreferenceExample]:
    """Generate synthetic preference examples."""
    
    examples = []
    
    for prompt in prompts:
        # Use base response if available, otherwise generate generic one
        if prompt in base_responses:
            base_response = base_responses[prompt]
        else:
            # Generate a generic response based on prompt type
            base_response = generate_generic_response(prompt)
        
        # Generate variations
        variations = generate_response_variations(prompt, base_response)
        
        if len(variations) >= 2:
            chosen = variations[0]  # Enhanced version
            rejected = random.choice(variations[1:])  # Random lower quality version
            
            # Determine quality difference type
            quality_types = ["detailed_vs_brief", "helpful_vs_vague", "accurate_vs_incorrect", "structured_vs_rambling"]
            quality_type = random.choice(quality_types)
            
            example = PreferenceExample(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                metadata={
                    "source": "synthetic",
                    "quality_type": quality_type,
                    "chosen_length": len(chosen.split()),
                    "rejected_length": len(rejected.split())
                }
            )
            examples.append(example)
    
    return examples

def generate_generic_response(prompt: str) -> str:
    """Generate a generic response for prompts without predefined answers."""
    
    prompt_lower = prompt.lower()
    
    if "how" in prompt_lower and ("learn" in prompt_lower or "improve" in prompt_lower):
        return f"To address this topic effectively, start with understanding the fundamentals and practice regularly. Break down the process into manageable steps, seek feedback from others, and be patient with your progress. Consistency and dedication are key to success in this area."
    
    elif "what" in prompt_lower and "best way" in prompt_lower:
        return f"The most effective approach involves careful planning and systematic execution. Consider your specific goals and constraints, research different methods, and choose the approach that best fits your situation. Don't be afraid to adapt your strategy based on what you learn along the way."
    
    elif "write" in prompt_lower and "story" in prompt_lower:
        return f"In a world not unlike our own, something extraordinary was about to happen. The protagonist faced a challenge that would change everything they thought they knew. Through courage, determination, and perhaps a little luck, they discovered that the greatest adventures often begin with a single step into the unknown."
    
    elif "explain" in prompt_lower:
        return f"This is an important concept that affects many aspects of our daily lives. The basic principle involves several interconnected elements working together to create a larger system. Understanding how these components interact helps us appreciate the complexity and elegance of the overall process."
    
    else:
        return f"This is a thoughtful question that deserves careful consideration. There are multiple perspectives to consider, each with their own merits and limitations. The key is to examine the available evidence, consider different viewpoints, and draw conclusions based on logical reasoning and reliable information."

def main():
    """Generate large synthetic preference dataset."""
    print("🤖 Generating Large Synthetic Preference Dataset")
    print("=" * 60)
    
    # Create diverse prompts
    prompts = create_diverse_prompts()
    print(f"📝 Created {len(prompts)} diverse prompts")
    
    # Create base responses for quality examples
    base_responses = create_base_responses()
    print(f"💬 Created {len(base_responses)} high-quality base responses")
    
    # Generate synthetic preferences
    print("🔄 Generating synthetic preference pairs...")
    examples = generate_synthetic_preferences(prompts, base_responses)
    
    # Add some variety by shuffling and sampling
    random.shuffle(examples)
    
    print(f"✅ Generated {len(examples)} synthetic preference examples")
    
    # Save the large dataset
    synthetic_data = [example.to_dict() for example in examples]
    
    with open("large_synthetic_preferences.json", "w") as f:
        json.dump(synthetic_data, f, indent=2)
    
    print(f"💾 Saved large synthetic dataset: large_synthetic_preferences.json")
    
    # Create quality breakdown
    quality_breakdown = {}
    for example in examples:
        quality_type = example.metadata.get("quality_type", "unknown")
        quality_breakdown[quality_type] = quality_breakdown.get(quality_type, 0) + 1
    
    print("\n📊 Quality Type Breakdown:")
    for quality_type, count in quality_breakdown.items():
        print(f"   {quality_type}: {count} examples")
    
    # Calculate average lengths
    chosen_lengths = [ex.metadata.get("chosen_length", 0) for ex in examples]
    rejected_lengths = [ex.metadata.get("rejected_length", 0) for ex in examples]
    
    print(f"\n📏 Response Length Stats:")
    print(f"   Average chosen response: {sum(chosen_lengths)/len(chosen_lengths):.1f} words")
    print(f"   Average rejected response: {sum(rejected_lengths)/len(rejected_lengths):.1f} words")
    
    print(f"\n🎯 Usage:")
    print("   python train_rlhf.py --method dpo \\")
    print("                        --model_path your_model.pth \\")
    print("                        --preference_data large_synthetic_preferences.json")
    
    print(f"\n🚀 Ready to train with {len(examples)} synthetic preferences!")

if __name__ == "__main__":
    main()