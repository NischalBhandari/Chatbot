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

    def chat(
        self,
        messages: list,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Chat with the model using a conversation format

        Args:
            messages: List of message dictionaries with 'role' and 'content'
                     Example: [{"role": "user", "content": "Hello!"}]
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated response
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call download_and_load_model() first.")

        # Try to use chat template if available
        try:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback to simple formatting
                prompt = ""
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    prompt += f"{role}: {content}\n"
                prompt += "assistant: "
        except Exception:
            # Fallback to simple formatting
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"{role}: {content}\n"
            prompt += "assistant: "

        return self.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )

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

    def interactive_chat(
        self,
        max_length: int = 200,
        temperature: float = 0.8,
        system_prompt: Optional[str] = None
    ):
        """
        Start an interactive chat session with the model

        Args:
            max_length: Maximum length of generated responses
            temperature: Sampling temperature (higher = more creative)
            system_prompt: Optional system prompt to set context
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call download_and_load_model() first.")

        print("\n" + "=" * 60)
        print("Interactive Chat Mode")
        print("=" * 60)
        print("\nCommands:")
        print("  - Type your message and press Enter to chat")
        print("  - Type 'quit', 'exit', or 'q' to exit")
        print("  - Type 'clear' to clear conversation history")
        print("  - Type 'info' to see model information")
        print("\nNote: Press Ctrl+C to interrupt generation\n")

        conversation_history = []

        # Add system prompt if provided
        if system_prompt:
            conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
            print(f"System: {system_prompt}\n")

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == 'clear':
                    conversation_history = []
                    if system_prompt:
                        conversation_history.append({
                            "role": "system",
                            "content": system_prompt
                        })
                    print("\nConversation history cleared.\n")
                    continue

                if user_input.lower() == 'info':
                    print("\nModel Info:")
                    print(json.dumps(self.get_model_info(), indent=2))
                    print()
                    continue

                if not user_input:
                    continue

                # Add user message to history
                conversation_history.append({
                    "role": "user",
                    "content": user_input
                })

                # Generate response
                print("\nAssistant: ", end="", flush=True)

                # Format conversation for generation
                try:
                    if hasattr(self.tokenizer, 'apply_chat_template'):
                        prompt = self.tokenizer.apply_chat_template(
                            conversation_history,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        # Simple fallback formatting
                        prompt = ""
                        for msg in conversation_history:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")
                            if role == "system":
                                prompt += f"System: {content}\n"
                            elif role == "user":
                                prompt += f"User: {content}\n"
                            elif role == "assistant":
                                prompt += f"Assistant: {content}\n"
                        prompt += "Assistant: "
                except Exception:
                    # Fallback formatting
                    prompt = ""
                    for msg in conversation_history:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        prompt += f"{role}: {content}\n"
                    prompt += "assistant: "

                # Generate response
                response = self.generate(
                    prompt,
                    max_length=len(self.tokenizer.encode(prompt)) + max_length,
                    temperature=temperature,
                    top_p=0.9
                )

                # Extract only the new response (remove the prompt)
                if response.startswith(prompt):
                    assistant_response = response[len(prompt):].strip()
                else:
                    # Fallback: try to extract after last occurrence of "assistant:"
                    parts = response.lower().split("assistant:")
                    if len(parts) > 1:
                        assistant_response = response.split("assistant:")[-1].strip()
                    else:
                        assistant_response = response.strip()

                # Clean up response (remove any trailing user prompts)
                for stop_word in ["\nUser:", "\nuser:", "\nYou:", "\nyou:"]:
                    if stop_word in assistant_response:
                        assistant_response = assistant_response.split(stop_word)[0].strip()

                print(assistant_response)
                print()

                # Add assistant response to history
                conversation_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })

            except KeyboardInterrupt:
                print("\n\nGeneration interrupted. Type 'quit' to exit.\n")
                # Remove the last user message since we didn't complete the response
                if conversation_history and conversation_history[-1]["role"] == "user":
                    conversation_history.pop()
            except Exception as e:
                print(f"\nError: {str(e)}\n")
                # Remove the last user message on error
                if conversation_history and conversation_history[-1]["role"] == "user":
                    conversation_history.pop()


def main():
    """
    Interactive chat mode with the LLM
    """
    print("\n" + "=" * 60)
    print("LLM Runner - Interactive Chat")
    print("=" * 60)

    # Available models
    print("\nRecommended models:")
    print("1. gpt2 - Lightest, fastest (500MB)")
    print("2. TinyLlama/TinyLlama-1.1B-Chat-v1.0 - Better quality (2GB)")
    print("3. microsoft/phi-2 - High quality (5GB, use 4-bit)")
    print("4. Custom - Enter any Hugging Face model name")

    choice = input("\nSelect model (1-4) [default: 1]: ").strip()

    # Map choice to model name
    model_map = {
        "1": "gpt2",
        "2": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "3": "microsoft/phi-2",
        "": "gpt2"  # default
    }

    if choice in model_map:
        model_name = model_map[choice]
        use_4bit = (choice == "3")  # Use 4-bit for phi-2
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

        # Ask for system prompt
        print("\n" + "-" * 60)
        use_system = input("Set a system prompt? (y/n) [default: n]: ").strip().lower()
        system_prompt = None

        if use_system == 'y':
            print("\nExample system prompts:")
            print("  - You are a helpful assistant.")
            print("  - You are a Python programming expert.")
            print("  - You are a creative story writer.")
            system_prompt = input("\nEnter system prompt: ").strip()

        # Start interactive chat
        runner.interactive_chat(
            max_length=200,
            temperature=0.8,
            system_prompt=system_prompt
        )
    else:
        print("\nFailed to load model. Please try again.")


if __name__ == "__main__":
    main()
