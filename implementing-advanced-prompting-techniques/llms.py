from langchain_openai import ChatOpenAI

def get_cot_llm():
    """LLM for Chain-of-Thought agent (more thoughtful, detailed responses)"""
    return ChatOpenAI(
        model_name="gpt-4",
        temperature=0.2,  # Lower for more deterministic reasoning
        max_tokens=2000   # Allow longer step-by-step explanations
    )

def get_react_llm():
    """LLM for ReAct agent (better at tool usage and action planning)"""
    return ChatOpenAI(
        model_name="gpt-4",
        temperature=0.3,  # Slightly higher for flexibility in tool use
        max_tokens=1000
    )

def get_fewshot_llm():
    """LLM for Few-shot agent (good at following examples)"""
    return ChatOpenAI(
        model_name="gpt-4-1106-preview",  # Better at in-context learning
        temperature=0.1,  # Very low to stick closely to examples
        max_tokens=1500
    )

def get_cot_fewshot_llm():
    """LLM for Combined CoT + Few-shot agent"""
    return ChatOpenAI(
        model_name="gpt-4",
        temperature=0.15,  # Balanced between creativity and structure
        max_tokens=2500    # Allow for both examples and reasoning
    )
