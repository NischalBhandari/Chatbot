"""
Example scripts for using the LLM Runner
"""
from llm_runner import LLMRunner
import json


def example_1_basic_generation():
    """
    Example 1: Basic text generation with GPT-2
    """
    print("=" * 60)
    print("Example 1: Basic Text Generation")
    print("=" * 60)

    # Use a small model for quick testing
    runner = LLMRunner("gpt2")

    # Download and load
    if runner.download_and_load_model():
        # Generate text
        prompt = "The future of artificial intelligence is"
        print(f"\nPrompt: {prompt}")

        result = runner.generate(
            prompt,
            max_length=100,
            temperature=0.7
        )

        print(f"\nGenerated:\n{result}")


def example_2_creative_writing():
    """
    Example 2: Creative writing with higher temperature
    """
    print("\n" + "=" * 60)
    print("Example 2: Creative Writing (High Temperature)")
    print("=" * 60)

    runner = LLMRunner("gpt2")

    if runner.download_and_load_model():
        prompt = "In a world where robots and humans coexist,"

        print(f"\nPrompt: {prompt}")

        result = runner.generate(
            prompt,
            max_length=150,
            temperature=1.0,  # Higher temperature = more creative
            top_p=0.95
        )

        print(f"\nGenerated:\n{result}")


def example_3_code_generation():
    """
    Example 3: Code generation (works better with code-trained models)
    """
    print("\n" + "=" * 60)
    print("Example 3: Code Generation")
    print("=" * 60)

    # For better code generation, use models like:
    # - "microsoft/phi-2"
    # - "codellama/CodeLlama-7b-hf"
    # For this example, we'll use gpt2 (limited capability)

    runner = LLMRunner("gpt2")

    if runner.download_and_load_model():
        prompt = "def fibonacci(n):\n    "

        print(f"\nPrompt:\n{prompt}")

        result = runner.generate(
            prompt,
            max_length=200,
            temperature=0.3  # Lower temperature for code
        )

        print(f"\nGenerated:\n{result}")


def example_4_chat_interface():
    """
    Example 4: Using chat interface
    """
    print("\n" + "=" * 60)
    print("Example 4: Chat Interface")
    print("=" * 60)

    # For better chat, use chat-tuned models like:
    # - "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # - "microsoft/phi-2"

    runner = LLMRunner("gpt2")

    if runner.download_and_load_model():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are three benefits of exercise?"}
        ]

        print("\nConversation:")
        for msg in messages:
            print(f"{msg['role']}: {msg['content']}")

        response = runner.chat(messages, max_length=200, temperature=0.7)

        print(f"\nResponse:\n{response}")


def example_5_quantized_model():
    """
    Example 5: Using a larger model with 4-bit quantization
    """
    print("\n" + "=" * 60)
    print("Example 5: Quantized Model (Memory Efficient)")
    print("=" * 60)

    # Using phi-2 with 4-bit quantization
    # This reduces memory usage significantly
    print("\nNote: This example uses microsoft/phi-2 (~5GB)")
    print("Set use_4bit=True to reduce memory usage\n")

    choice = input("Download and run phi-2? (y/n): ").lower()

    if choice == 'y':
        runner = LLMRunner("microsoft/phi-2", use_4bit=True)

        if runner.download_and_load_model():
            # Show model info
            print("\nModel Info:")
            print(json.dumps(runner.get_model_info(), indent=2))

            prompt = "Explain the concept of machine learning in simple terms:"

            print(f"\nPrompt: {prompt}")

            result = runner.generate(
                prompt,
                max_length=300,
                temperature=0.7
            )

            print(f"\nGenerated:\n{result}")
    else:
        print("Skipped.")


def example_6_model_comparison():
    """
    Example 6: Compare different generation parameters
    """
    print("\n" + "=" * 60)
    print("Example 6: Parameter Comparison")
    print("=" * 60)

    runner = LLMRunner("gpt2")

    if runner.download_and_load_model():
        prompt = "The best way to learn programming is"

        print(f"\nPrompt: {prompt}\n")

        # Low temperature (more focused)
        print("\n--- Low Temperature (0.3) - Focused ---")
        result1 = runner.generate(prompt, max_length=80, temperature=0.3)
        print(result1)

        # Medium temperature
        print("\n--- Medium Temperature (0.7) - Balanced ---")
        result2 = runner.generate(prompt, max_length=80, temperature=0.7)
        print(result2)

        # High temperature (more creative)
        print("\n--- High Temperature (1.2) - Creative ---")
        result3 = runner.generate(prompt, max_length=80, temperature=1.2)
        print(result3)


def example_7_interactive_mode():
    """
    Example 7: Interactive chat mode
    """
    print("\n" + "=" * 60)
    print("Example 7: Interactive Mode")
    print("=" * 60)

    runner = LLMRunner("gpt2")

    if runner.download_and_load_model():
        print("\nInteractive mode. Type 'quit' to exit.")
        print("Note: GPT-2 is not fine-tuned for chat, so responses may vary.\n")

        conversation_history = []

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Add to conversation
            conversation_history.append({
                "role": "user",
                "content": user_input
            })

            # Generate response
            try:
                response = runner.chat(
                    conversation_history,
                    max_length=150,
                    temperature=0.8
                )

                # Extract just the new part
                print(f"\nAssistant: {response}\n")

                # Add to history
                conversation_history.append({
                    "role": "assistant",
                    "content": response
                })

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}\n")


def main():
    """
    Run examples
    """
    print("\n🤖 LLM Runner Examples\n")
    print("Choose an example to run:")
    print("1. Basic text generation")
    print("2. Creative writing (high temperature)")
    print("3. Code generation")
    print("4. Chat interface")
    print("5. Quantized model (larger model, memory efficient)")
    print("6. Parameter comparison")
    print("7. Interactive chat mode")
    print("0. Run all examples (1-4)")

    choice = input("\nEnter choice (0-7): ").strip()

    examples = {
        "1": example_1_basic_generation,
        "2": example_2_creative_writing,
        "3": example_3_code_generation,
        "4": example_4_chat_interface,
        "5": example_5_quantized_model,
        "6": example_6_model_comparison,
        "7": example_7_interactive_mode,
    }

    if choice == "0":
        example_1_basic_generation()
        example_2_creative_writing()
        example_3_code_generation()
        example_4_chat_interface()
    elif choice in examples:
        examples[choice]()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
