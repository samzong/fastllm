#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from typing import List, AsyncGenerator
import time
import json
from contextlib import asynccontextmanager

import torch
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from pydantic import BaseModel
from huggingface_hub import configure_http_backend

# 配置参数
class Config:
    HOST = "127.0.0.1"       # 服务监听地址
    PORT = 8000           # 服务监听端口
    USE_HF_MIRROR = True  # 是否使用 HuggingFace 镜像
    GPU_COUNT = 1         # 使用的 GPU 数量
    MODEL_NAME = None      # 模型名称，将通过命令行参数设置

# 全局变量存储LLM模型实例
llm_model = None

def get_available_gpu_count() -> int:
    """获取可用的 GPU 数量"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

def configure_hf_mirror():
    """配置 HuggingFace 镜像站点"""
    # 设置环境变量（兼容性方案）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # 使用官方推荐方式配置HTTP后端
    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.proxies = {
            "http": "https://hf-mirror.com",
            "https": "https://hf-mirror.com"
        }
        return session
    
    configure_http_backend(backend_factory=backend_factory)
    print("📡 已配置 HuggingFace 镜像: hf-mirror.com")

def download_model(model_name: str, use_hf_mirror: bool = True) -> str:
    """加载模型，支持本地路径或HuggingFace模型 """
    # 最简单的判断：如果路径存在，就是本地模型
    if os.path.exists(model_name):
        print(f"📁 使用本地模型: {model_name}")
        return model_name
    
    # 否则从HuggingFace下载
    from huggingface_hub import snapshot_download
    
    # 配置镜像
    if use_hf_mirror:
        configure_hf_mirror()
    
    # 使用模型名称作为目录名
    model_dir = os.path.join(os.getcwd(), model_name.split("/")[-1])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    try:
        print(f"📥 从 HuggingFace 下载: {model_name}")
        model_path = snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            max_workers=4, 
            local_dir_use_symlinks=False
        )
        print(f"✅ 下载完成: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        raise

def load_model(model_path: str, gpu_count: int = 1) -> LLM:
    """加载模型到VLLM"""
    available_gpus = get_available_gpu_count()
    if available_gpus > 0:
        print(f"🖥️ 检测到GPU: {available_gpus}个")
        gpu_count = min(gpu_count, available_gpus)
    else:
        print("⚠️ 未检测到GPU，将使用CPU模式")
        gpu_count = 1
    
    print(f"🔄 加载模型中...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=gpu_count,
        trust_remote_code=True,
        dtype="auto"
    )
    print(f"✅ 模型加载完成")
    return llm

@asynccontextmanager
async def lifespan(app: FastAPI):
    """处理应用的生命周期事件"""
    global llm_model
    model_path = download_model(Config.MODEL_NAME, Config.USE_HF_MIRROR)
    llm_model = load_model(model_path, Config.GPU_COUNT)
    print(f"🚀 服务就绪")
    yield
    if llm_model:
        del llm_model
        torch.cuda.empty_cache()
        print("🛑 服务已停止")

# 创建FastAPI应用
app = FastAPI(title="LLM API Service", lifespan=lifespan)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 数据模型
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI兼容的聊天完成接口"""
    if not llm_model:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    
    # 构建提示文本
    prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n"
    
    prompt += "Assistant: "
    
    # 如果请求流式输出
    if request.stream:
        return StreamingResponse(
            stream_response(prompt, request),
            media_type="text/event-stream"
        )
    
    # 非流式输出处理
    outputs = llm_model.generate(
        [prompt],
        SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
    )
    
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": outputs[0].outputs[0].text},
            "finish_reason": "stop"
        }]
    }

async def stream_response(prompt: str, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """生成流式响应"""
    request_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())
    
    # 创建请求参数
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    
    # 提交生成请求
    results_generator = llm_model.generate_async(
        [prompt],
        sampling_params,
    )
    
    # 流式输出标头
    yield "data: " + json.dumps({
        'id': request_id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': request.model,
        'choices': [{
            'index': 0,
            'delta': {'role': 'assistant'},
            'finish_reason': None
        }]
    }) + "\n\n"
    
    # 追踪之前发送的内容，用于计算增量
    previous_text = ""
    
    async for result in results_generator:
        output: RequestOutput = result[0]
        if output.outputs:
            current_text = output.outputs[0].text
            # 计算增量文本
            delta_text = current_text[len(previous_text):]
            previous_text = current_text
            
            # 发送增量更新
            yield "data: " + json.dumps({
                'id': request_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': request.model,
                'choices': [{
                    'index': 0,
                    'delta': {'content': delta_text},
                    'finish_reason': None
                }]
            }) + "\n\n"
    
    # 发送结束标记
    yield "data: " + json.dumps({
        'id': request_id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': request.model,
        'choices': [{
            'index': 0,
            'delta': {},
            'finish_reason': 'stop'
        }]
    }) + "\n\n"
    
    # 标准的SSE结束标记
    yield "data: [DONE]\n\n"

@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "data": [{"id": Config.MODEL_NAME.split("/")[-1]}],
        "object": "list"
    }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "model_loaded": llm_model is not None}

def main():
    """主函数"""
    # 获取可用 GPU 数量
    available_gpus = get_available_gpu_count()
    default_gpu_count = min(available_gpus, 1) if available_gpus > 0 else 1
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='OpenAI 兼容的 LLM API 服务器',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 位置参数
    parser.add_argument(
        'model_name', 
        type=str,
        help='模型名称或路径，例如:\n'
             '- HF模型: Qwen/Qwen2-0.5B-Instruct\n'
             '- 本地模型: ./models/my-model\n'
             '注: 存在路径视为本地模型，否则从HF下载'
    )
    
    # 可选参数组
    group = parser.add_argument_group('可选参数')
    
    group.add_argument(
        '--gpu-count', '-g',
        type=int, 
        default=default_gpu_count,
        help=f'使用的GPU数量 (默认: {default_gpu_count})'
    )
    
    group.add_argument(
        '--host', 
        type=str, 
        default=Config.HOST,
        help=f'监听地址 (默认: {Config.HOST})'
    )
    
    group.add_argument(
        '--port', '-p',
        type=int, 
        default=Config.PORT,
        help=f'监听端口 (默认: {Config.PORT})'
    )
    
    group.add_argument(
        '--no-hf-mirror', 
        action='store_true',
        help='不使用HF镜像 (默认: 使用hf-mirror.com)'
    )
    
    group.add_argument(
        '--download-only', '-d',
        action='store_true',
        help='仅下载模型，不启动服务'
    )
    
    args = parser.parse_args()
    
    # 更新配置
    Config.MODEL_NAME = args.model_name
    Config.GPU_COUNT = args.gpu_count
    Config.HOST = args.host
    Config.PORT = args.port
    Config.USE_HF_MIRROR = not args.no_hf_mirror
    
    # 处理仅下载模式
    if args.download_only:
        print(f"📥 下载模式 | 模型: {Config.MODEL_NAME}")
        if Config.USE_HF_MIRROR:
            print("   使用HF镜像: hf-mirror.com")
        
        try:
            model_path = download_model(Config.MODEL_NAME, Config.USE_HF_MIRROR)
            print(f"✅ 模型下载成功: {model_path}")
            return
        except Exception as e:
            print(f"❌ 模型下载失败: {e}")
            return
    
    # 启动服务器模式
    print(f"🚀 启动服务 | 模型: {Config.MODEL_NAME}")
    print(f"   地址: {Config.HOST}:{Config.PORT} | GPU: {Config.GPU_COUNT}")
    if Config.USE_HF_MIRROR:
        print("   使用HF镜像: hf-mirror.com")
    
    # 启动服务
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)

if __name__ == "__main__":
    main() 