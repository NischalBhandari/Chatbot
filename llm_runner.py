"""
LLM Runner - Download and run Large Language Models from Hugging Face
"""
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from typing import Optional, Dict, Any
import json


class LLMRunner:
    """
    A class to download and run LLMs from Hugging Face
    """

    def __init__(self, model_name: str, use_8bit: bool = False, use_4bit: bool = False):
        """
        Initialize the LLM Runner

        Args:
            model_name: Hugging Face model identifier (e.g., "meta-llama/Llama-2-7b-chat-hf")
            use_8bit: Use 8-bit quantization to reduce memory usage
            use_4bit: Use 4-bit quantization to reduce memory usage even more
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit

        print(f"Device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    def download_and_load_model(self, cache_dir: Optional[str] = None):
        """
        Download and load the model from Hugging Face

        Args:
            cache_dir: Optional directory to cache the downloaded model
        """
        print(f"\nDownloading and loading model: {self.model_name}")
        print("This may take a while depending on the model size...")

        try:
            # Load tokenizer
            print("\nLoading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )

            # Set pad token if not already set (fixes attention mask warning)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Configure quantization if requested
            quantization_config = None
            if self.use_4bit:
                print("Using 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.use_8bit:
                print("Using 8-bit quantization...")
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            # Load model
            print("\nLoading model...")
            model_kwargs = {
                "cache_dir": cache_dir,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            # Move to device if not using quantization
            if not quantization_config and self.device == "cuda":
                self.model = self.model.to(self.device)

            print(f"\n✓ Model loaded successfully!")
            return True

        except Exception as e:
            print(f"\n✗ Error loading model: {str(e)}")
            return False

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> str:
        """
        Generate text from a prompt

        Args:
            prompt: The input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to use sampling or greedy decoding

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call download_and_load_model() first.")

        print("\nGenerating response...")

        # Tokenize input with attention mask
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        if self.device == "cuda" and not (self.use_4bit or self.use_8bit):
            inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        info = {
            "model_name": self.model_name,
            "device": self.device,
            "quantization": "4-bit" if self.use_4bit else "8-bit" if self.use_8bit else "none",
        }

        if hasattr(self.model, 'num_parameters'):
            info["num_parameters"] = self.model.num_parameters()

        return info

    def interactive_generation(
        self,
        max_length: int = 200,
        temperature: float = 0.8
    ):
        """
        Interactive text generation mode - provide prompts and get completions

        Args:
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more creative)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call download_and_load_model() first.")

        print("\n" + "=" * 60)
        print("Interactive Text Generation Mode")
        print("=" * 60)
        print("\nThis mode generates text continuations from your prompts.")
        print("\nCommands:")
        print("  - Type your prompt and press Enter to generate")
        print("  - Type 'quit', 'exit', or 'q' to exit")
        print("  - Type 'info' to see model information")
        print("\nNote: Press Ctrl+C to interrupt generation\n")

        while True:
            try:
                # Get user input
                user_input = input("Prompt: ").strip()

                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == 'info':
                    print("\nModel Info:")
                    print(json.dumps(self.get_model_info(), indent=2))
                    print()
                    continue

                if not user_input:
                    continue

                # Generate continuation
                print("\nGenerated text:")
                print("-" * 60)

                response = self.generate(
                    user_input,
                    max_length=len(self.tokenizer.encode(user_input)) + max_length,
                    temperature=temperature,
                    top_p=0.9
                )

                print(response)
                print("-" * 60)
                print()

            except KeyboardInterrupt:
                print("\n\nGeneration interrupted.\n")
            except Exception as e:
                print(f"\nError: {str(e)}\n")


def main():
    """
    Interactive text generation with the LLM
    """
    print("\n" + "=" * 60)
    print("LLM Runner - Text Generation")
    print("=" * 60)

    # Available models
    print("\nRecommended models:")
    print("1. gpt2 - Lightest, fastest (500MB) ⭐ RECOMMENDED")
    print("2. gpt2-medium - Better quality (1.5GB)")
    print("3. gpt2-large - High quality (3GB)")
    print("4. Custom - Enter any Hugging Face model name")

    choice = input("\nSelect model (1-4) [default: 1]: ").strip()

    # Map choice to model name
    model_map = {
        "1": "gpt2",
        "2": "gpt2-medium",
        "3": "gpt2-large",
        "": "gpt2"  # default
    }

    if choice in model_map:
        model_name = model_map[choice]
        use_4bit = False
    elif choice == "4":
        model_name = input("Enter Hugging Face model name: ").strip()
        quantize = input("Use 4-bit quantization? (y/n) [default: n]: ").strip().lower()
        use_4bit = (quantize == 'y')
    else:
        print("Invalid choice. Using gpt2.")
        model_name = "gpt2"
        use_4bit = False

    print(f"\nInitializing {model_name}...")

    # Initialize runner
    runner = LLMRunner(model_name, use_4bit=use_4bit)

    # Download and load model
    if runner.download_and_load_model():
        # Show model info
        print("\nModel Info:")
        print(json.dumps(runner.get_model_info(), indent=2))

        # Start interactive generation
        print("\n" + "-" * 60)
        print("Starting interactive text generation...")
        print("Tip: Give it a sentence or paragraph to continue")

        runner.interactive_generation(
            max_length=200,
            temperature=0.8
        )
    else:
        print("\nFailed to load model. Please try again.")


if __name__ == "__main__":
    main()
