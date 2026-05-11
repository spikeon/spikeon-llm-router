from router import chat
from config import MODELS

def print_help():
    print("\nCommands:")
    print("  /model <key>  → force model (snappy/fast/coder/balanced/thinker/smart)")
    print("  /models       → list all models + speeds")
    print("  /clear        → clear conversation history")
    print("  /verbose      → toggle routing info")
    print("  /exit         → quit\n")

def print_models():
    print("\nAvailable models:")
    for key, data in MODELS.items():
        print(f"  {key:<10} {data['tokens_per_sec']:>4} t/s  →  {data['name']}")
    print()

def main():
    print("Local AI Router")
    print("Type /help for commands\n")

    history = []
    verbose = False
    force_model = None

    while True:
        try:
            user_input = input("you: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_input:
            continue

        # commands
        if user_input.startswith("/"):
            cmd = user_input.split()

            if cmd[0] == "/help":
                print_help()

            elif cmd[0] == "/models":
                print_models()

            elif cmd[0] == "/model" and len(cmd) > 1:
                key = cmd[1]
                if key in MODELS:
                    force_model = key
                    print(f"→ locked to: {key}")
                else:
                    print(f"→ unknown model. choose: {list(MODELS.keys())}")

            elif cmd[0] == "/model" and len(cmd) == 1:
                force_model = None
                print("→ model lock cleared, back to auto-routing")

            elif cmd[0] == "/clear":
                history = []
                print("→ history cleared")

            elif cmd[0] == "/verbose":
                verbose = not verbose
                print(f"→ verbose: {'on' if verbose else 'off'}")

            elif cmd[0] == "/exit":
                print("Bye.")
                break

            else:
                print(f"→ unknown command. type /help")

            continue

        # chat
        try:
            response, model_used = chat(
                prompt=user_input,
                override_model=force_model,
                conversation_history=history,
                verbose=verbose
            )

            # update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

            print(f"\nai ({model_used}): {response}\n")

        except Exception as e:
            print(f"→ error: {e}")

if __name__ == "__main__":
    main()