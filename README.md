# FastLLM

A minimal LLM server launcher in just ~100 lines of Python code.

## Features

- 🚀 **Simple**: Launch an OpenAI-compatible LLM API server with a single command
- 📦 **Flexible**: Works with both local models and models from HuggingFace
- ⚡ **Fast**: Includes optimization options for faster loading and inference
- 🔄 **Interruptible**: Clean Ctrl+C handling for easy operation
- 🌐 **HF Mirror**: Built-in support for HuggingFace mirror (for faster downloads in some regions)

## Requirements

- Python 3.8+
- vLLM library installed (`pip install vllm`)
- GPU with CUDA support (recommended)

## Usage

### Basic Usage

```bash
# Start an LLM server with a local model
python llm_server.py ./models/my-local-model

# Download and run a model from HuggingFace
python llm_server.py Qwen/Qwen2-7B-Instruct

# Download without starting the server
python llm_server.py Qwen/Qwen2-7B-Instruct --download-only
```

### Optimization Options

```bash
# Run with half-precision for faster loading on low-end GPUs
python llm_server.py Qwen/Qwen2-7B-Instruct --dtype half

# Use quantization to reduce memory usage
python llm_server.py Qwen/Qwen2-7B-Instruct --quantization awq

# Use safetensors format for faster loading
python llm_server.py Qwen/Qwen2-7B-Instruct --load-format safetensors

# Use multiple GPUs
python llm_server.py Qwen/Qwen2-7B-Instruct --gpu-count 2
```

### Server Configuration

```bash
# Change host and port
python llm_server.py Qwen/Qwen2-7B-Instruct --host 0.0.0.0 --port 8080
```

## API Usage

Once the server is running, you can use it as an OpenAI-compatible API endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-7B-Instruct",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "temperature": 0.7
  }'
```

Or with Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="Qwen/Qwen2-7B-Instruct",
    messages=[{"role": "user", "content": "Tell me a joke"}]
)

print(response.choices[0].message.content)
```

## How It Works

This tool is a thin wrapper around vLLM's `serve` command, adding convenient features like:

1. Automatic model downloading from HuggingFace
2. Simple configuration of optimization parameters
3. Clean interruption handling
4. Helpful status messages

All in less than 100 lines of core code!

## 📝 License

MIT License

Copyright (c) 2024 samzong

See the [LICENSE](LICENSE) file for details.