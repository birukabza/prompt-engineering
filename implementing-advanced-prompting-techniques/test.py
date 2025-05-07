import json
from chains import build_all

with open("data.json", "r", encoding="utf-8") as f:
    questions = json.load(f)


def test_agents(agents, questions):
    results = {"cot": [], "react": [], "fewshot": [], "cot_fewshot": []}

    for question_data in questions:
        question = question_data["question"]
        correct_answer = question_data["answer"]

        for agent_name, agent in agents.items():
            print(f"Testing {agent_name} with question: {question}")
            response = agent.run(question)
            results[agent_name].append(
                {
                    "question": question,
                    "expected_answer": correct_answer,
                    "agent_answer": response,
                    "correct": response.strip() == correct_answer.strip(),
                }
            )

    return results


if __name__ == "__main__":
    agents = build_all()

    test_results = test_agents(agents, questions)

    for agent_name, agent_results in test_results.items():
        print(f"\nResults for {agent_name}:")
        for result in agent_results:
            print(f"Q: {result['question']}")
            print(f"Expected: {result['expected_answer']}")
            print(f"Agent Answer: {result['agent_answer']}")
            print(f"Correct: {result['correct']}")
            print("-" * 50)
