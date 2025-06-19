"""
注册表初始化 - 将所有组件注册到系统中
"""

import logging
from pathlib import Path

# 导入注册表
from .core.model_registry import ModelRegistry
from .core.quantizer_registry import QuantizerRegistry
from .core.outlier_suppression_registry import OutlierSuppressionRegistry

# 导入模型适配器
from .models.base_adapter import BaseHuggingFaceAdapter
from .models.llama_adapter import LlamaAdapter
from .models.qwen_adapter import QwenAdapter

# 导入量化器
from .quantizers.gptq_quantizer import GPTQQuantizer
from .quantizers.awq_quantizer import AWQQuantizer
from .quantizers.minmax_quantizer import MinMaxQuantizer

# 导入离群值抑制器
from .outlier_suppression.smooth_quant import SmoothQuantSuppressor

logger = logging.getLogger(__name__)

def register_all_components():
    """注册所有组件"""
    logger.info("注册所有组件...")
    
    # 注册模型适配器
    register_model_adapters()
    
    # 注册量化器
    register_quantizers()
    
    # 注册离群值抑制器
    register_outlier_suppressors()
    
    logger.info("组件注册完成")

def register_model_adapters():
    """注册模型适配器"""
    logger.info("注册模型适配器...")
    
    # 注册默认适配器
    ModelRegistry.register("default", BaseHuggingFaceAdapter)
    
    # 注册Llama适配器
    ModelRegistry.register(
        "llama", 
        LlamaAdapter, 
        patterns=["llama", "Llama", "meta-llama", "llama-2", "llama-3"]
    )
    
    # 注册Qwen适配器
    ModelRegistry.register(
        "qwen", 
        QwenAdapter, 
        patterns=["qwen", "Qwen", "Qwen2", "Qwen-1.5", "Qwen-2"]
    )
    
    # 注册DeepSeek适配器（使用默认适配器）
    ModelRegistry.register(
        "deepseek", 
        BaseHuggingFaceAdapter, 
        patterns=["deepseek", "DeepSeek", "deepseek-ai"]
    )

def register_quantizers():
    """注册量化器"""
    logger.info("注册量化器...")
    
    # 注册GPTQ量化器
    QuantizerRegistry.register("gptq", GPTQQuantizer)
    
    # 注册AWQ量化器
    QuantizerRegistry.register("awq", AWQQuantizer)
    
    # 注册MinMax量化器
    QuantizerRegistry.register("minmax", MinMaxQuantizer)
    
    # 注册HQQ量化器（占位符）
    # QuantizerRegistry.register("hqq", HQQQuantizer)

def register_outlier_suppressors():
    """注册离群值抑制器"""
    logger.info("注册离群值抑制器...")
    
    # 注册SmoothQuant抑制器
    OutlierSuppressionRegistry.register("smooth_quant", SmoothQuantSuppressor)

def get_registered_components():
    """获取已注册的组件信息"""
    return {
        "models": ModelRegistry.list_adapters(),
        "quantizers": QuantizerRegistry.list_quantizers(),
        "outlier_suppressors": OutlierSuppressionRegistry.list_suppressors()
    }

def print_registered_components():
    """打印已注册的组件信息"""
    components = get_registered_components()
    
    print("=" * 50)
    print("已注册的组件:")
    print("=" * 50)
    
    print("模型适配器:")
    for model in components["models"]:
        print(f"  - {model}")
    
    print("\n量化器:")
    for quantizer in components["quantizers"]:
        info = QuantizerRegistry.get_quantizer_info(quantizer)
        print(f"  - {quantizer}")
        print(f"    支持的位数: {info['supported_bits']}")
        print(f"    支持的组大小: {info['supported_group_sizes']}")
    
    print("\n离群值抑制器:")
    for suppressor in components["outlier_suppressors"]:
        info = OutlierSuppressionRegistry.get_suppressor_info(suppressor)
        print(f"  - {suppressor}")
        print(f"    支持的模型: {info['supported_models']}")
    
    print("=" * 50)

# 自动注册所有组件
register_all_components() 