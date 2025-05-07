from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent

from llms import get_cot_llm, get_react_llm, get_cot_fewshot_llm, get_fewshot_llm
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Few-shot examples for calculus tutoring
FEW_SHOT_EXAMPLES = [
    {
        "input": "Find the derivative of x^2",
        "output": "Let's solve this step by step:\n1. Identify the power rule: d/dx(x^n) = n*x^(n-1)\n2. Apply to x^2: 2*x^(2-1)\n3. Final answer: 2x",
    },
    {
        "input": "What's the integral of 3x^2?",
        "output": "Let's solve this step by step:\n1. Identify the power rule for integrals: ∫x^n dx = x^(n+1)/(n+1) + C\n2. Apply to 3x^2: 3*(x^3/3) + C\n3. Simplify: x^3 + C",
    },
]


def build_fewshot_agent():
    fewshot_llm = get_fewshot_llm()  

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=FEW_SHOT_EXAMPLES,
        embeddings=OpenAIEmbeddings(),
        vectorstore_cls=FAISS,
        k=2,
    )

    fewshot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate(
            input_variables=["input", "output"],
            template="Input: {input}\nOutput: {output}",
        ),
        prefix="You are a calculus tutor. Here are some examples:",
        suffix="Now solve this new problem:\nInput: {input}\nOutput:",
        input_variables=["input"],
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=5, return_messages=True
    )

    return LLMChain(llm=fewshot_llm, prompt=fewshot_prompt, memory=memory, verbose=True)


def build_cot_fewshot_agent():
    cot_fewshot_llm = get_cot_fewshot_llm()

    combined_prompt = """You are an expert calculus tutor. Follow these steps:
1. Recall similar problems you've seen before
2. Apply Chain-of-Thought reasoning
3. Provide a complete solution

Here are some examples:
"input": "What's the integral of 3x^2?",
        "output": "Let's solve this step by step:\n1. Identify the power rule for integrals: ∫x^n dx = x^(n+1)/(n+1) + C\n2. Apply to 3x^2: 3*(x^3/3) + C\n3. Simplify: x^3 + C"
"input": "Find the derivative of x^2",
        "output": "Let's solve this step by step:\n1. Identify the power rule: d/dx(x^n) = n*x^(n-1)\n2. Apply to x^2: 2*x^(2-1)\n3. Final answer: 2x"
Now solve this problem using the same approach:
{input}

Let's think step-by-step:
"""

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=FEW_SHOT_EXAMPLES,
        embeddings=OpenAIEmbeddings(),
        vectorstore_cls=FAISS,
        k=2,
    )

    prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate(
            input_variables=["input", "output"],
            template="Input: {input}\nOutput: {output}",
        ),
        prefix=combined_prompt,
        suffix="",
        input_variables=["input"],
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=5, return_messages=True
    )

    return LLMChain(
        llm=cot_fewshot_llm, prompt=prompt_template, memory=memory, verbose=True
    )


_COT_PROMPT = """
You are an expert calculus tutor helping a student understand and solve calculus problems.

When answering, follow these steps:
1) Restate the problem in your own words
2) Identify relevant calculus concepts or theorems
3) Outline a step-by-step approach
4) Execute each step with clear reasoning and intermediate results
5) Summarize the final answer and verify it

Conversation History:
{chat_history}

Student's Question:
{input}

Let's think step-by-step:
"""



def build_cot_chain():
    cot_llm = get_cot_llm()
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=5, return_messages=True
    )

    cot_chain = LLMChain(
        llm=cot_llm,
        prompt=PromptTemplate(
            input_variables=["chat_history", "input"], template=_COT_PROMPT
        ),
        memory=memory,
        verbose=True,
    )
    return cot_chain




def build_react_agent():
    react_llm = get_react_llm()
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=5, return_messages=True
    )
    wolfram = Tool(
        name="WolframAlpha",
        func=WolframAlphaAPIWrapper().run,
        description="Computational knowledge engine for math and science",
    )
    tools = [wolfram]
    agent = initialize_agent(
        tools=tools,
        llm=react_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        max_iterations=5,
    )
    return agent


def build_all():
    return {
        "cot": build_cot_chain(),
        "react": build_react_agent(),
        "fewshot": build_fewshot_agent(),
        "cot_fewshot": build_cot_fewshot_agent(),
    }
