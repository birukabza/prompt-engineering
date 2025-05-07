from dotenv import load_dotenv

from chains import build_all

load_dotenv()


def select_chain(chains):
    """
    Prompt the user to select one of the available tutor chains.
    """
    options = list(chains.keys())
    print("Available tutoring modes:")
    for idx, name in enumerate(options, start=1):
        print(f"  {idx}. {name.title()} Tutor")
    while True:
        choice = input("Enter the number of your choice: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid selection. Please enter a valid number.")


def format_response(response):
    """Format the agent's response for beautiful display"""
    if isinstance(response, dict):
        text = response.get('text', str(response))
    else:
        text = str(response)
    
    formatted = []
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith(('1)', '2)', '3)', '4)', '5)', 'a)', 'b)', 'c)')):
            formatted.append(f"  {line}")
        elif line:
            formatted.append(line)
    
    return '\n'.join(formatted)

def chat_loop(agent):
    """
    Engage in a chat session with the provided agent.
    """
    print("\nType 'exit' at any time to end the session.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit", "bye"}:
            print("\nTutor: Goodbye! Keep practicing.\n")
            break
        try:
            response = agent.invoke({"input": query})
            if response.get("output"):
                reply = format_response(response["output"])
            else:
                reply = format_response(response)

        except Exception as e:
            reply = f"An error occurred: {e}"
        
        print("\n" + "="*50)
        print("Tutor's Response:")
        print("-"*50)
        print(reply)
        print("="*50 + "\n")




def main():
    print("\n===== Welcome to the Calculus Tutor =====\n")

    tutors = build_all()

    mode = select_chain(tutors)
    agent = tutors[mode]
    print(f"\nStarting chat with the '{mode.title()}' tutor...\n")

    chat_loop(agent)


if __name__ == "__main__":
    main()
