from eval_harness.judge import grade_answer_with_llm
from sklearn.metrics import cohen_kappa_score

# Simulated 10 test cases where the HUMAN has already graded the answers.
# 1 = Bad/Hallucination, 5 = Perfect.
test_cases = [
    {"q": "What color is the sky?", "c": "The sky is blue.", "a": "The sky is blue.", "human": 5},
    {"q": "What color is the sky?", "c": "The sky is blue.", "a": "The sky is green.", "human": 1},
    {"q": "Who wrote Hamlet?", "c": "William Shakespeare wrote Hamlet.", "a": "Shakespeare wrote it.", "human": 5},
    {"q": "Who wrote Hamlet?", "c": "William Shakespeare wrote Hamlet.", "a": "Charles Dickens wrote Hamlet.", "human": 1},
    {"q": "What is water made of?", "c": "Water is H2O.", "a": "Water is made of hydrogen and oxygen.", "human": 5},
    {"q": "What is water made of?", "c": "Water is H2O.", "a": "Water is made of carbon dioxide.", "human": 1},
    {"q": "When was the moon landing?", "c": "The moon landing was in 1969.", "a": "It happened in 1969.", "human": 5},
    {"q": "When was the moon landing?", "c": "The moon landing was in 1969.", "a": "It happened in 1985.", "human": 1},
    {"q": "What is the boiling point of water?", "c": "Water boils at 100 degrees Celsius.", "a": "100 degrees Celsius.", "human": 5},
    {"q": "What is the boiling point of water?", "c": "Water boils at 100 degrees Celsius.", "a": "It boils at 50 degrees.", "human": 1},
]

def validate_judge():
    human_scores = []
    llm_scores = []
    
    print(f"Validating judge on {len(test_cases)} cases...")
    
    for case in test_cases:
        result = grade_answer_with_llm(case['q'], case['c'], case['a'])
        
        # Convert the 1-5 score into a binary "Good" (1) or "Bad" (0)
        human_binary = 1 if case['human'] >= 4 else 0
        llm_binary = 1 if result['score'] >= 4 else 0
        
        human_scores.append(human_binary)
        llm_scores.append(llm_binary)
        
        print(f"Q: {case['q']} | Human: {case['human']} | LLM: {result['score']}")

    # Calculate Inter-Rater Reliability
    kappa = cohen_kappa_score(human_scores, llm_scores)
    agreement = sum(1 for h, l in zip(human_scores, llm_scores) if h == l) / len(human_scores) * 100
    
    print("\n" + "="*40)
    print(f"Raw Agreement: {agreement:.1f}%")
    print(f"Cohen's Kappa: {kappa:.3f}")
    print("="*40)

if __name__ == "__main__":
    validate_judge()