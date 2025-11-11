"""
Quick start script for interactive chat with LLMs
"""
from llm_runner import LLMRunner
import sys


def quick_chat(model_name: str = "gpt2", use_4bit: bool = False):
    """
    Quick start function for chatting with an LLM

    Args:
        model_name: The Hugging Face model to use
        use_4bit: Whether to use 4-bit quantization
    """
    print(f"\nStarting chat with {model_name}...")

    # Initialize and load model
    runner = LLMRunner(model_name, use_4bit=use_4bit)

    if runner.download_and_load_model():
        # Start interactive chat
        runner.interactive_chat(max_length=200, temperature=0.8)
    else:
        print("\nFailed to load model.")


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        model = sys.argv[1]
        quantize = "--4bit" in sys.argv or "-4" in sys.argv
        quick_chat(model, quantize)
    else:
        # Default: use gpt2
        print("Usage: python chat.py [model_name] [--4bit]")
        print("\nExamples:")
        print("  python chat.py")
        print("  python chat.py gpt2")
        print("  python chat.py TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print("  python chat.py microsoft/phi-2 --4bit")
        print("\nStarting with default model (gpt2)...\n")
        quick_chat()
