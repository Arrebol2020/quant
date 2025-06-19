#!/usr/bin/env python3
"""
框架测试脚本
"""

import sys
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.registry_init import print_registered_components
from src.core.model_registry import ModelRegistry
from src.core.quantizer_registry import QuantizerRegistry
from src.core.outlier_suppression_registry import OutlierSuppressionRegistry
from src.utils.logging import setup_logging

def test_registry():
    """测试注册表功能"""
    print("=" * 60)
    print("测试注册表功能")
    print("=" * 60)
    
    # 打印已注册的组件
    print_registered_components()
    
    # 测试获取量化器信息
    print("\n测试量化器信息:")
    for quantizer_name in QuantizerRegistry.list_quantizers():
        try:
            info = QuantizerRegistry.get_quantizer_info(quantizer_name)
            print(f"  {quantizer_name}: {info}")
        except Exception as e:
            print(f"  获取 {quantizer_name} 信息失败: {e}")
    
    # 测试获取离群值抑制器信息
    print("\n测试离群值抑制器信息:")
    for suppressor_name in OutlierSuppressionRegistry.list_suppressors():
        try:
            info = OutlierSuppressionRegistry.get_suppressor_info(suppressor_name)
            print(f"  {suppressor_name}: {info}")
        except Exception as e:
            print(f"  获取 {suppressor_name} 信息失败: {e}")

def test_model_adapters():
    """测试模型适配器"""
    print("\n" + "=" * 60)
    print("测试模型适配器")
    print("=" * 60)
    
    # 测试路径匹配
    test_paths = [
        "/path/to/llama-2-7b",
        "/path/to/qwen-1.5-7b",
        "/path/to/deepseek-7b",
        "/path/to/unknown-model"
    ]
    
    for path in test_paths:
        try:
            adapter = ModelRegistry.get_adapter(Path(path))
            print(f"  {path} -> {adapter.__class__.__name__}")
        except Exception as e:
            print(f"  {path} -> 错误: {e}")

def test_config_loading():
    """测试配置加载"""
    print("\n" + "=" * 60)
    print("测试配置加载")
    print("=" * 60)
    
    from src.utils.config import load_config, get_default_config
    
    # 测试默认配置
    default_config = get_default_config()
    print("默认配置:")
    for key, value in default_config.items():
        print(f"  {key}: {value}")
    
    # 测试配置文件加载
    config_files = [
        "configs/gptq_config.yaml",
        "configs/awq_config.yaml",
        "configs/smooth_quant_config.yaml"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            try:
                config = load_config(config_file)
                print(f"\n{config_file} 加载成功:")
                print(f"  量化方法: {config.get('quantization', {}).get('method', 'N/A')}")
            except Exception as e:
                print(f"{config_file} 加载失败: {e}")
        else:
            print(f"{config_file} 文件不存在")

def main():
    """主测试函数"""
    # 设置日志
    setup_logging(logging.INFO)
    
    print("大模型量化工具框架测试")
    print("=" * 60)
    
    try:
        # 测试注册表
        test_registry()
        
        # 测试模型适配器
        test_model_adapters()
        
        # 测试配置加载
        test_config_loading()
        
        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 