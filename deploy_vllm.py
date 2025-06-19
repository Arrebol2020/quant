#!/usr/bin/env python3
"""
vLLM部署示例脚本
演示如何使用量化工具量化的模型在vLLM上部署
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Optional

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_quantization_method(model_path: Path) -> Optional[str]:
    """从模型路径或配置文件推断量化方法"""
    # 检查vLLM配置文件
    vllm_config_path = model_path / "vllm_config.json"
    if vllm_config_path.exists():
        try:
            with open(vllm_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('quantization', {}).get('quantization_method')
        except Exception as e:
            logging.warning(f"读取vLLM配置文件失败: {e}")
    
    # 检查模型配置
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('quantization_config', {}).get('quantization_method')
        except Exception as e:
            logging.warning(f"读取模型配置失败: {e}")
    
    # 根据路径名推断
    model_path_str = str(model_path)
    if 'gptq' in model_path_str.lower():
        return 'gptq'
    elif 'awq' in model_path_str.lower():
        return 'awq'
    elif 'minmax' in model_path_str.lower():
        return 'minmax'
    
    return None

def generate_vllm_command(model_path: str, quantization: Optional[str] = None, 
                         port: int = 8000, host: str = "0.0.0.0", 
                         tensor_parallel_size: int = 1):
    """生成vLLM启动命令"""
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"模型路径不存在: {model_path}")
    
    # 如果没有指定量化方法，尝试自动推断
    if quantization is None:
        quantization = get_quantization_method(model_path)
        if quantization:
            logging.info(f"自动检测到量化方法: {quantization}")
        else:
            logging.warning("无法自动检测量化方法，将使用默认设置")
    
    # 生成vLLM命令
    cmd_parts = [
        "python -m vllm.entrypoints.openai.api_server",
        f"--model {model_path}",
        f"--port {port}",
        f"--host {host}",
        f"--tensor-parallel-size {tensor_parallel_size}"
    ]
    
    if quantization:
        cmd_parts.append(f"--quantization {quantization}")
    
    return " ".join(cmd_parts)

def generate_python_example(model_path: str, quantization: Optional[str] = None):
    """生成Python API使用示例"""
    
    if quantization is None:
        quantization = get_quantization_method(Path(model_path)) or "auto"
    
    example_code = f'''# vLLM Python API 使用示例
from vllm import LLM, SamplingParams

# 加载量化模型
llm = LLM(
    model="{model_path}",
    quantization="{quantization}" if "{quantization}" != "auto" else None
)

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100
)

# 生成文本
prompts = [
    "Hello, how are you?",
    "What is the capital of France?",
    "Explain quantum computing in simple terms."
]

outputs = llm.generate(prompts, sampling_params)

# 打印结果
for i, output in enumerate(outputs):
    print(f"Prompt {{i+1}}: {{prompts[i]}}")
    print(f"Response: {{output.outputs[0].text}}")
    print("-" * 50)
'''
    
    return example_code

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="vLLM部署工具")
    parser.add_argument("--model-path", required=True, help="量化模型路径")
    parser.add_argument("--quantization", help="量化方法 (gptq/awq/minmax)")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="张量并行大小")
    parser.add_argument("--generate-command", action="store_true", help="生成vLLM启动命令")
    parser.add_argument("--generate-example", action="store_true", help="生成Python示例代码")
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        # 检查模型路径
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"错误: 模型路径不存在: {model_path}")
            return
        
        print("=" * 60)
        print("vLLM部署工具")
        print("=" * 60)
        print(f"模型路径: {model_path}")
        
        # 自动检测量化方法
        detected_quantization = get_quantization_method(model_path)
        if detected_quantization:
            print(f"检测到的量化方法: {detected_quantization}")
        
        # 生成vLLM启动命令
        if args.generate_command:
            print("\n" + "=" * 60)
            print("vLLM启动命令:")
            print("=" * 60)
            
            cmd = generate_vllm_command(
                args.model_path, 
                args.quantization or detected_quantization,
                args.port,
                args.host,
                args.tensor_parallel_size
            )
            print(cmd)
        
        # 生成Python示例
        if args.generate_example:
            print("\n" + "=" * 60)
            print("Python API 使用示例:")
            print("=" * 60)
            
            example = generate_python_example(
                args.model_path,
                args.quantization or detected_quantization
            )
            print(example)
        
        # 如果没有指定任何选项，显示所有信息
        if not args.generate_command and not args.generate_example:
            print("\n" + "=" * 60)
            print("vLLM启动命令:")
            print("=" * 60)
            
            cmd = generate_vllm_command(
                args.model_path, 
                args.quantization or detected_quantization,
                args.port,
                args.host,
                args.tensor_parallel_size
            )
            print(cmd)
            
            print("\n" + "=" * 60)
            print("Python API 使用示例:")
            print("=" * 60)
            
            example = generate_python_example(
                args.model_path,
                args.quantization or detected_quantization
            )
            print(example)
        
        print("\n" + "=" * 60)
        print("部署说明:")
        print("=" * 60)
        print("1. 确保已安装vLLM: pip install vllm")
        print("2. 使用上述命令启动vLLM服务")
        print("3. 服务启动后，可以通过HTTP API或Python API使用")
        print("4. HTTP API地址: http://localhost:8000/v1")
        print("5. 支持OpenAI兼容的API调用")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 