#!/usr/bin/env python3
"""
简单的框架结构测试
"""

import os
import sys
from pathlib import Path

def test_directory_structure():
    """测试目录结构"""
    print("=" * 60)
    print("测试目录结构")
    print("=" * 60)
    
    # 检查主要目录
    directories = [
        "src",
        "src/core",
        "src/models", 
        "src/quantizers",
        "src/outlier_suppression",
        "src/utils",
        "configs",
        "docs"
    ]
    
    for directory in directories:
        if Path(directory).exists():
            print(f"✓ {directory} 存在")
        else:
            print(f"✗ {directory} 不存在")
    
    # 检查主要文件
    files = [
        "quantize.py",
        "requirements.txt",
        "README.md",
        "src/__init__.py",
        "src/core/__init__.py",
        "src/core/quantization_manager.py",
        "src/core/model_registry.py",
        "src/core/quantizer_registry.py",
        "src/core/outlier_suppression_registry.py",
        "src/models/base_adapter.py",
        "src/quantizers/gptq_quantizer.py",
        "src/quantizers/awq_quantizer.py",
        "src/quantizers/minmax_quantizer.py",
        "src/outlier_suppression/smooth_quant.py",
        "src/utils/config.py",
        "src/utils/logging.py",
        "configs/gptq_config.yaml",
        "configs/awq_config.yaml",
        "configs/minmax_config.yaml",
        "configs/smooth_quant_config.yaml",
        "docs/extension_guide.md"
    ]
    
    print("\n检查主要文件:")
    for file in files:
        if Path(file).exists():
            print(f"✓ {file} 存在")
        else:
            print(f"✗ {file} 不存在")

def test_import_structure():
    """测试导入结构"""
    print("\n" + "=" * 60)
    print("测试导入结构")
    print("=" * 60)
    
    # 添加src目录到Python路径
    src_path = Path("src")
    if src_path.exists():
        sys.path.insert(0, str(src_path.absolute()))
    
    try:
        # 测试核心模块导入
        print("测试核心模块导入...")
        from core.quantization_manager import QuantizationManager
        print("✓ QuantizationManager 导入成功")
        
        from core.model_registry import ModelRegistry
        print("✓ ModelRegistry 导入成功")
        
        from core.quantizer_registry import QuantizerRegistry
        print("✓ QuantizerRegistry 导入成功")
        
        from core.outlier_suppression_registry import OutlierSuppressionRegistry
        print("✓ OutlierSuppressionRegistry 导入成功")
        
    except ImportError as e:
        print(f"✗ 核心模块导入失败: {e}")
    
    try:
        # 测试模型适配器导入
        print("\n测试模型适配器导入...")
        from models.base_adapter import BaseHuggingFaceAdapter
        print("✓ BaseHuggingFaceAdapter 导入成功")
        
        from models.llama_adapter import LlamaAdapter
        print("✓ LlamaAdapter 导入成功")
        
        from models.qwen_adapter import QwenAdapter
        print("✓ QwenAdapter 导入成功")
        
    except ImportError as e:
        print(f"✗ 模型适配器导入失败: {e}")
    
    try:
        # 测试量化器导入
        print("\n测试量化器导入...")
        from quantizers.gptq_quantizer import GPTQQuantizer
        print("✓ GPTQQuantizer 导入成功")
        
        from quantizers.awq_quantizer import AWQQuantizer
        print("✓ AWQQuantizer 导入成功")
        
        from quantizers.minmax_quantizer import MinMaxQuantizer
        print("✓ MinMaxQuantizer 导入成功")
        
    except ImportError as e:
        print(f"✗ 量化器导入失败: {e}")
    
    try:
        # 测试离群值抑制器导入
        print("\n测试离群值抑制器导入...")
        from outlier_suppression.smooth_quant import SmoothQuantSuppressor
        print("✓ SmoothQuantSuppressor 导入成功")
        
    except ImportError as e:
        print(f"✗ 离群值抑制器导入失败: {e}")
    
    try:
        # 测试工具模块导入
        print("\n测试工具模块导入...")
        from utils.config import load_config, get_default_config
        print("✓ 配置工具导入成功")
        
        from utils.logging import setup_logging
        print("✓ 日志工具导入成功")
        
    except ImportError as e:
        print(f"✗ 工具模块导入失败: {e}")

def test_registry_functionality():
    """测试注册表功能"""
    print("\n" + "=" * 60)
    print("测试注册表功能")
    print("=" * 60)
    
    try:
        # 导入注册表
        from core.model_registry import ModelRegistry
        from core.quantizer_registry import QuantizerRegistry
        from core.outlier_suppression_registry import OutlierSuppressionRegistry
        
        # 导入组件
        from models.base_adapter import BaseHuggingFaceAdapter
        from models.llama_adapter import LlamaAdapter
        from models.qwen_adapter import QwenAdapter
        from quantizers.gptq_quantizer import GPTQQuantizer
        from quantizers.awq_quantizer import AWQQuantizer
        from quantizers.minmax_quantizer import MinMaxQuantizer
        from outlier_suppression.smooth_quant import SmoothQuantSuppressor
        
        # 注册组件
        print("注册组件...")
        
        # 注册模型适配器
        ModelRegistry.register("default", BaseHuggingFaceAdapter)
        ModelRegistry.register("llama", LlamaAdapter, patterns=["llama"])
        ModelRegistry.register("qwen", QwenAdapter, patterns=["qwen"])
        
        # 注册量化器
        QuantizerRegistry.register("gptq", GPTQQuantizer)
        QuantizerRegistry.register("awq", AWQQuantizer)
        QuantizerRegistry.register("minmax", MinMaxQuantizer)
        
        # 注册离群值抑制器
        OutlierSuppressionRegistry.register("smooth_quant", SmoothQuantSuppressor)
        
        print("✓ 组件注册成功")
        
        # 测试列表功能
        print(f"\n已注册的模型适配器: {ModelRegistry.list_adapters()}")
        print(f"已注册的量化器: {QuantizerRegistry.list_quantizers()}")
        print(f"已注册的离群值抑制器: {OutlierSuppressionRegistry.list_suppressors()}")
        
        # 测试获取功能
        print("\n测试获取功能...")
        adapter = ModelRegistry.get_adapter(Path("/path/to/llama-model"))
        print(f"✓ 获取模型适配器成功: {adapter.__class__.__name__}")
        
        quantizer = QuantizerRegistry.get_quantizer("gptq")
        print(f"✓ 获取GPTQ量化器成功: {quantizer.__class__.__name__}")
        
        quantizer = QuantizerRegistry.get_quantizer("awq")
        print(f"✓ 获取AWQ量化器成功: {quantizer.__class__.__name__}")
        
        quantizer = QuantizerRegistry.get_quantizer("minmax")
        print(f"✓ 获取MinMax量化器成功: {quantizer.__class__.__name__}")
        
        suppressor = OutlierSuppressionRegistry.get_suppressor("smooth_quant")
        print(f"✓ 获取离群值抑制器成功: {suppressor.__class__.__name__}")
        
        # 测试量化器信息
        print("\n测试量化器信息...")
        for quantizer_name in ["gptq", "awq", "minmax"]:
            try:
                info = QuantizerRegistry.get_quantizer_info(quantizer_name)
                print(f"✓ {quantizer_name}: 支持位数={info['supported_bits']}, 支持组大小={info['supported_group_sizes']}")
            except Exception as e:
                print(f"✗ 获取{quantizer_name}信息失败: {e}")
        
    except Exception as e:
        print(f"✗ 注册表功能测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_minmax_quantizer():
    """测试MinMax量化器"""
    print("\n" + "=" * 60)
    print("测试MinMax量化器")
    print("=" * 60)
    
    try:
        from quantizers.minmax_quantizer import MinMaxQuantizer
        
        # 创建MinMax量化器
        quantizer = MinMaxQuantizer()
        print(f"✓ MinMax量化器创建成功: {quantizer.name}")
        
        # 测试支持的位数
        supported_bits = quantizer.get_supported_bits()
        print(f"✓ 支持的位数: {supported_bits}")
        
        # 测试支持的组大小
        supported_group_sizes = quantizer.get_supported_group_sizes()
        print(f"✓ 支持的组大小: {supported_group_sizes}")
        
        # 测试量化参数计算
        import torch
        
        # 创建测试权重
        test_weight = torch.randn(10, 10)
        print(f"✓ 测试权重创建成功: shape={test_weight.shape}")
        
        # 测试全局参数计算
        scale, zero_point = quantizer._compute_global_params(test_weight, bits=8, symmetric=False)
        print(f"✓ 全局参数计算成功: scale={scale:.4f}, zero_point={zero_point:.4f}")
        
        # 测试对称参数计算
        scale_sym, zero_point_sym = quantizer._compute_global_params(test_weight, bits=8, symmetric=True)
        print(f"✓ 对称参数计算成功: scale={scale_sym:.4f}, zero_point={zero_point_sym:.4f}")
        
        # 测试量化权重
        quantized_weight = quantizer._quantize_weight_global_minmax(test_weight, scale, zero_point, bits=8)
        print(f"✓ 权重量化成功: shape={quantized_weight.shape}")
        
        print("✓ MinMax量化器测试完成")
        
    except Exception as e:
        print(f"✗ MinMax量化器测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("大模型量化工具框架 - 简单测试")
    print("=" * 60)
    
    # 测试目录结构
    test_directory_structure()
    
    # 测试导入结构
    test_import_structure()
    
    # 测试注册表功能
    test_registry_functionality()
    
    # 测试MinMax量化器
    test_minmax_quantizer()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    main() 