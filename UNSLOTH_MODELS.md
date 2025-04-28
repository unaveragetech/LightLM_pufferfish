# Unsloth Pre-Quantized Models

This document provides a comprehensive list of pre-quantized models available through the Unsloth library for efficient fine-tuning. All these models are optimized for training with 4-bit quantization, which significantly reduces memory usage and increases training speed.

For the most up-to-date list, visit the [Unsloth HuggingFace page](https://huggingface.co/unsloth).

## How to Use These Models

You can use any of these models in the training interface by selecting "Custom model name" and entering the model path (e.g., `unsloth/vera-7b-instruct-bnb-4bit`).

## Model Categories

### Recommended Models

| Model | Size | Description | Link |
|-------|------|-------------|------|
| **unsloth/mistral-7b-instruct-v0.3-bnb-4bit** | 7B | High-quality instruction-tuned model with good balance of performance and quality | [Link](https://huggingface.co/unsloth/mistral-7b-instruct-v0.3-bnb-4bit) |
| **unsloth/llama-3.1-8b-instruct-bnb-4bit** | 8B | Meta's LLaMA 3.1 model with excellent instruction-following and reasoning | [Link](https://huggingface.co/unsloth/llama-3.1-8b-instruct-bnb-4bit) |
| **unsloth/llama-3.2-1b-instruct-bnb-4bit** | 1B | Smaller LLaMA 3.2 model for testing or resource-constrained environments | [Link](https://huggingface.co/unsloth/llama-3.2-1b-instruct-bnb-4bit) |
| **unsloth/gemma-3-4b-it-qat-bnb-4bit** | 4B | Google's Gemma 3 model with image capabilities and strong multimodal performance | [Link](https://huggingface.co/unsloth/gemma-3-4b-it-qat-bnb-4bit) |

### High-Quality Models

| Model | Size | Description | Link |
|-------|------|-------------|------|
| **unsloth/phi-4-14b-instruct-bnb-4bit** | 14B | Microsoft's Phi-4 model with excellent reasoning and instruction-following | [Link](https://huggingface.co/unsloth/phi-4-14b-instruct-bnb-4bit) |
| **unsloth/qwen2.5-7b-instruct-bnb-4bit** | 7B | Alibaba's Qwen 2.5 model with strong multilingual capabilities | [Link](https://huggingface.co/unsloth/qwen2.5-7b-instruct-bnb-4bit) |
| **unsloth/vera-7b-instruct-bnb-4bit** | 7B | Vera model optimized for reasoning and strong performance on logical and mathematical tasks | [Link](https://huggingface.co/unsloth/vera-7b-instruct-bnb-4bit) |
| **unsloth/deepseek-r1-bnb-4bit** | 7B | DeepSeek R1 model optimized for research and complex tasks | [Link](https://huggingface.co/unsloth/deepseek-r1-bnb-4bit) |

### Small/Fast Models

| Model | Size | Description | Link |
|-------|------|-------------|------|
| **unsloth/tinyllama-1.1b-chat-v1.0-bnb-4bit** | 1.1B | Extremely small and fast model ideal for testing or very limited resources | [Link](https://huggingface.co/unsloth/tinyllama-1.1b-chat-v1.0-bnb-4bit) |
| **unsloth/phi-2-bnb-4bit** | 2.7B | Microsoft's Phi-2 model, smaller but still capable | [Link](https://huggingface.co/unsloth/phi-2-bnb-4bit) |
| **unsloth/stablelm-2-1.6b-zephyr-bnb-4bit** | 1.6B | Small StableLM model fine-tuned on Zephyr | [Link](https://huggingface.co/unsloth/stablelm-2-1.6b-zephyr-bnb-4bit) |

### Specialized Models

| Model | Size | Description | Link |
|-------|------|-------------|------|
| **unsloth/codellama-7b-instruct-bnb-4bit** | 7B | Specialized for code generation and understanding | [Link](https://huggingface.co/unsloth/codellama-7b-instruct-bnb-4bit) |
| **unsloth/mistral-7b-openorca-bnb-4bit** | 7B | Mistral fine-tuned on OpenOrca dataset for improved reasoning | [Link](https://huggingface.co/unsloth/mistral-7b-openorca-bnb-4bit) |
| **unsloth/neural-chat-7b-v3-3-bnb-4bit** | 7B | Optimized for conversational AI applications | [Link](https://huggingface.co/unsloth/neural-chat-7b-v3-3-bnb-4bit) |
| **unsloth/solar-10.7b-instruct-v1.0-bnb-4bit** | 10.7B | Upstage's SOLAR model with strong instruction-following | [Link](https://huggingface.co/unsloth/solar-10.7b-instruct-v1.0-bnb-4bit) |

### Multilingual Models

| Model | Size | Description | Link |
|-------|------|-------------|------|
| **unsloth/qwen2.5-7b-instruct-bnb-4bit** | 7B | Strong multilingual capabilities, especially for Chinese | [Link](https://huggingface.co/unsloth/qwen2.5-7b-instruct-bnb-4bit) |
| **unsloth/yi-1.5-9b-chat-bnb-4bit** | 9B | Yi model with good multilingual support | [Link](https://huggingface.co/unsloth/yi-1.5-9b-chat-bnb-4bit) |
| **unsloth/mistral-7b-instruct-v0.3-bnb-4bit** | 7B | Good performance across multiple languages | [Link](https://huggingface.co/unsloth/mistral-7b-instruct-v0.3-bnb-4bit) |

## Hardware Requirements

The memory requirements for these models vary based on their size:

- **1-3B parameter models**: Minimum 8GB VRAM (e.g., GTX 1070)
- **7-8B parameter models**: Minimum 12GB VRAM (e.g., RTX 3060)
- **10-14B parameter models**: Minimum 24GB VRAM (e.g., RTX 3090/4090)

### Precision Requirements

Some models require specific precision settings:

- **BF16 Required Models**: Vera, Gemma-3, and Phi-3 models require BF16 precision
- **FP16 Compatible Models**: Most other models work well with FP16 precision
- **Windows Users**: Set UNSLOTH_DISABLE_COMPILE=1 environment variable for compatibility

## Additional Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Unsloth HuggingFace Organization](https://huggingface.co/unsloth)
- [Unsloth Discord Community](https://discord.gg/unsloth)
