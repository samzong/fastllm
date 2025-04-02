#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import subprocess
import signal

def setup_signal_handler():
    """Setup signal handler for interruption"""
    def force_exit(sig, frame):
        print("\n🛑 Operation interrupted")
        os._exit(1)
    
    signal.signal(signal.SIGINT, force_exit)
    signal.signal(signal.SIGTERM, force_exit)

def download_model(model_name, use_hf_mirror=True):
    """Download model from HuggingFace if not a local path"""
    if os.path.exists(model_name):
        print(f"📁 Using local model: {model_name}")
        return model_name
    
    from huggingface_hub import snapshot_download
    
    model_dir = os.path.join(os.getcwd(), model_name.split("/")[-1])
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"📥 Downloading from HuggingFace: {model_name}")
    try:
        model_path = snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            max_workers=8,
            endpoint="https://hf-mirror.com" if use_hf_mirror else None,
            tqdm_class=None
        )
        print(f"\n✅ Download complete: {model_path}")
        return model_path
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        raise

def main():
    setup_signal_handler()
    
    parser = argparse.ArgumentParser(description='Simple LLM server launcher')
    parser.add_argument('model_name', type=str, help='Model name or path')
    
    # Basic options
    parser.add_argument('--host', type=str, default="127.0.0.1", help='Host address (default: 127.0.0.1)')
    parser.add_argument('--port', '-p', type=int, default=8000, help='Port number (default: 8000)')
    parser.add_argument('--no-hf-mirror', action='store_true', help='Disable HuggingFace mirror')
    parser.add_argument('--download-only', '-d', action='store_true', help='Download model only, do not start server')
    
    # Performance options
    perf_group = parser.add_argument_group('Performance')
    perf_group.add_argument('--gpu-count', '-g', type=int, default=1, help='Number of GPUs (default: 1)')
    perf_group.add_argument('--dtype', type=str, default="auto", choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'], 
                          help='Data type (default: auto), half/float16 for faster loading on low-end GPUs')
    perf_group.add_argument('--quantization', type=str, default=None, choices=[None, 'awq', 'gptq'],
                          help='Quantization method (default: None), AWQ/GPTQ reduces memory usage')
    perf_group.add_argument('--load-format', type=str, default="auto", choices=['auto', 'safetensors', 'gguf', 'dummy'],
                          help='Model loading format (default: auto), safetensors is typically fastest')
    
    args = parser.parse_args()
    
    try:
        if args.download_only:
            print(f"📥 Download mode | Model: {args.model_name}")
            if not args.no_hf_mirror:
                print("   Using HF mirror: hf-mirror.com")
            download_model(args.model_name, not args.no_hf_mirror)
            return
        
        # Download model if needed
        model_path = args.model_name
        if not os.path.exists(model_path):
            model_path = download_model(model_path, not args.no_hf_mirror)
        
        # Build vLLM command
        cmd = [
            "vllm", "serve",
            model_path,
            "--host", args.host,
            "--port", str(args.port),
            "--tensor-parallel-size", str(args.gpu_count),
            "--dtype", args.dtype
        ]
        
        # Add optional parameters
        if args.quantization:
            cmd.extend(["--quantization", args.quantization])
        
        if args.load_format != "auto":
            cmd.extend(["--load-format", args.load_format])
            
        # Display startup info
        print(f"🚀 Starting vLLM server | Model: {model_path}")
        print(f"   Address: {args.host}:{args.port} | GPUs: {args.gpu_count}")
        if args.dtype != "auto":
            print(f"   Data type: {args.dtype}")
        if args.quantization:
            print(f"   Quantization: {args.quantization}")
        if args.load_format != "auto":
            print(f"   Load format: {args.load_format}")
            
        # Start vLLM server
        result = subprocess.run(cmd)
        print("✅ Server exited normally" if result.returncode == 0 else f"❌ Server exited with code: {result.returncode}")
        
    except Exception as e:
        print(f"❌ Operation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
