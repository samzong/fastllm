# FastLLM - 轻量级大语言模型测试服务

一个轻量级、单文件的本地大语言模型 API 服务器，专为快速测试和原型开发而设计。提供 OpenAI 兼容接口，让你在本地环境中快速测试和开发 LLM 应用。

## ✨ 核心特性

- 🚀 **一键启动**：单文件实现，零配置即可启动服务
- 🔍 **快速原型**：适合快速验证想法和开发原型
- 📊 **流式响应**：支持类似 ChatGPT 的实时输出效果
- 🧩 **即插即用**：完全兼容 OpenAI API，可直接替换现有调用
- 🌐 **国内友好**：内置 HuggingFace 镜像，无需科学上网
- 📥 **简易部署**：支持本地模型和一键下载 HuggingFace 模型

## ⚡ 快速开始

### 1. 安装依赖

```bash
pip install vllm fastapi uvicorn torch huggingface_hub
```

### 2. 启动服务

```bash
# 使用 HuggingFace 模型
python llm_server.py Qwen/Qwen2-0.5B-Instruct

# 使用本地模型
python llm_server.py ./models/my-model
```

### 3. 测试接口

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-0.5B-Instruct",
    "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
    "temperature": 0.7
  }'
```

## 🔧 命令行参数

```
python llm_server.py [-h] [--gpu-count GPU_COUNT] [--host HOST] [--port PORT] [--no-hf-mirror] [--download-only] model_name
```

| 参数 | 说明 |
|------|------|
| `model_name` | 模型名称或路径 (必填) |
| `--gpu-count`, `-g` | 使用的GPU数量 |
| `--host` | 监听地址 (默认: 127.0.0.1) |
| `--port`, `-p` | 监听端口 (默认: 8000) |
| `--no-hf-mirror` | 不使用HuggingFace镜像 |
| `--download-only`, `-d` | 仅下载模型，不启动服务 |

## 💡 开发示例

### 标准调用

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen2-0.5B",
        "messages": [
            {"role": "user", "content": "你好，请简要介绍一下你自己"}
        ]
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### 流式调用

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen2-0.5B",
        "messages": [
            {"role": "user", "content": "写一篇短文章介绍人工智能"}
        ],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            if line == 'data: [DONE]':
                break
            data = json.loads(line[6:])
            delta = data['choices'][0].get('delta', {}).get('content', '')
            if delta:
                print(delta, end='', flush=True)
print()
```

## 📋 API 参考

| 端点 | 方法 | 描述 |
|------|------|------|
| `/v1/chat/completions` | POST | 聊天完成接口，支持流式输出 |
| `/v1/models` | GET | 获取可用模型 |
| `/health` | GET | 健康检查 |

## 🚀 适用场景

- 快速测试各种开源 LLM 模型效果
- 原型项目开发和验证
- 本地 AI 应用开发调试
- 教学和学习 LLM API 调用
- 无需联网的离线演示


## 📝 许可证

MIT License

Copyright (c) 2024 samzong

详见 [LICENSE](LICENSE) 文件 