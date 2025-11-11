# Basic LLM Runner

A Python project to download and run Large Language Models (LLMs) from Hugging Face.

## Features

- Download any LLM from Hugging Face
- Support for quantization (4-bit and 8-bit) to reduce memory usage
- Simple API for text generation
- Chat interface support
- GPU acceleration (CUDA) when available
- Configurable generation parameters

## Installation

1. Install Python 3.8 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: If you have an NVIDIA GPU, install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Basic Usage

```python
from llm_runner import LLMRunner

# Initialize with a model from Hugging Face
runner = LLMRunner("gpt2")

# Download and load the model
runner.download_and_load_model()

# Generate text
prompt = "Once upon a time"
result = runner.generate(prompt, max_length=100)
print(result)
```

### Using Quantization (for larger models)

```python
# Use 4-bit quantization to reduce memory usage
runner = LLMRunner("microsoft/phi-2", use_4bit=True)
runner.download_and_load_model()

result = runner.generate("Explain quantum computing:", max_length=200)
print(result)
```

### Chat Interface

```python
runner = LLMRunner("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
runner.download_and_load_model()

messages = [
    {"role": "user", "content": "What is Python?"}
]

response = runner.chat(messages, max_length=200)
print(response)
```

## Available Models

Check [config.json](config.json) for a list of recommended models:

- **gpt2**: Small model, good for testing (~500MB)
- **microsoft/phi-2**: Excellent 2.7B parameter model (~5GB)
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0**: Small chat model (~2GB)
- **mistralai/Mistral-7B-Instruct-v0.2**: Powerful 7B model (~14GB)

You can use any model from [Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation) by providing its identifier.

## Generation Parameters

Customize text generation with these parameters:

- `max_length`: Maximum length of generated text (default: 512)
- `temperature`: Controls randomness (0.0-2.0, default: 0.7)
  - Lower values = more focused/deterministic
  - Higher values = more creative/random
- `top_p`: Nucleus sampling (0.0-1.0, default: 0.9)
- `top_k`: Top-k sampling (default: 50)
- `do_sample`: Use sampling vs greedy decoding (default: True)

Example:
```python
result = runner.generate(
    "Write a story about",
    max_length=500,
    temperature=0.9,  # More creative
    top_p=0.95
)
```

## Memory Requirements

- **CPU only**: Can run smaller models (gpt2, phi-2 with quantization)
- **GPU with 4GB VRAM**: Small models or quantized medium models
- **GPU with 8GB VRAM**: Medium models (up to 3B parameters) with quantization
- **GPU with 16GB+ VRAM**: Large models (7B+) with quantization

## Quantization

Quantization reduces memory usage:

- **4-bit**: ~75% memory reduction, minimal quality loss
- **8-bit**: ~50% memory reduction, negligible quality loss

```python
# 4-bit quantization (most memory efficient)
runner = LLMRunner("mistralai/Mistral-7B-Instruct-v0.2", use_4bit=True)

# 8-bit quantization (balanced)
runner = LLMRunner("microsoft/phi-2", use_8bit=True)
```

## Examples

See [examples.py](examples.py) for more usage examples.

## Troubleshooting

### Out of Memory Error

- Try using quantization (`use_4bit=True`)
- Use a smaller model
- Reduce `max_length` parameter
- Close other applications

### Model Access Error

Some models (like Llama 2) require access approval:
1. Go to the model page on Hugging Face
2. Accept the terms and conditions
3. Login using: `huggingface-cli login`

### Slow Generation

- First generation is always slower (model loading)
- Use GPU if available
- Use smaller models for faster inference

## License

MIT License

## Resources

- [Hugging Face Models](https://huggingface.co/models)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
