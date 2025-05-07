# Calculus Tutor

This project implements and evaluates various prompt-engineering techniques for a calculus tutoring system using LangChain and OpenAI.

## Features

* **Few-Shot Prompting**: Uses semantic similarity to select relevant examples.
* **Chain-of-Thought (CoT)**: Guides the model through step-by-step reasoning.
* **Hybrid CoT + Few-Shot**: Combines example-driven guidance with CoT for highest accuracy.
* **ReAct Agent**: Integrates WolframAlpha for external computations.
* **Conversational Memory**: Maintains context with a sliding window of past interactions.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/calculus-tutor.git
   cd calculus-tutor
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key in `.env`:

   ```bash
   echo "OPENAI_API_KEY=your_key_here" > .env
   ```

## Usage

Run the tutor:

```bash
python main.py
```

Select a tutoring mode (cot, fewshot, cot\_fewshot, react) and start asking calculus questions interactively.

## License

MIT License
