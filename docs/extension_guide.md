# 扩展开发指南

本指南将帮助你扩展大模型量化工具，添加新的量化算法、模型支持和离群值抑制算法，并确保vLLM兼容性。

## 目录结构

```
src/
├── core/                    # 核心框架
│   ├── model_registry.py           # 模型注册表
│   ├── quantizer_registry.py       # 量化器注册表
│   └── outlier_suppression_registry.py  # 离群值抑制注册表
├── models/                  # 模型适配器
│   ├── base_adapter.py            # 基础适配器
│   ├── llama_adapter.py           # Llama适配器
│   └── qwen_adapter.py            # Qwen适配器
├── quantizers/             # 量化器实现
│   ├── gptq_quantizer.py          # GPTQ量化器
│   ├── awq_quantizer.py           # AWQ量化器
│   └── minmax_quantizer.py        # MinMax量化器
├── outlier_suppression/    # 离群值抑制器
│   └── smooth_quant.py            # SmoothQuant
└── utils/                  # 工具模块
    ├── config.py                  # 配置管理
    └── logging.py                 # 日志管理
```

## vLLM兼容性

为了确保量化的模型能在vLLM上直接部署，所有量化器都需要实现vLLM兼容性：

### 1. vLLM兼容性要求

- **量化参数**: 必须保存 `qweight`, `scales`, `zeros` 等属性
- **配置信息**: 在模型配置中添加 `quantization_config`
- **层属性**: 每个量化层需要包含量化方法、位数、组大小等信息

### 2. 实现vLLM兼容性

```python
def _add_vllm_compatibility(self, model, bits: int, group_size: int):
    """为模型添加vLLM兼容性"""
    # 添加配置信息
    if hasattr(model, 'config'):
        model.config.quantization_config = {
            "quantization_method": "your_method",
            "bits": bits,
            "group_size": group_size,
            # 其他特定参数
        }
    
    # 为所有量化层添加必要属性
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            module.qweight = module.weight.data
            module.scales = getattr(module, 'scale', torch.tensor(1.0))
            module.zeros = getattr(module, 'zero_point', torch.tensor(0.0))
            module.bits = bits
            module.group_size = group_size
            module.quantization_method = "your_method"
```

## 添加新的量化算法

### 1. 创建量化器类

在 `src/quantizers/` 目录下创建新的量化器文件，例如 `hqq_quantizer.py`：

```python
"""
HQQ量化器实现 - vLLM兼容版本
"""

import logging
import torch
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from ..core.quantizer_registry import BaseQuantizer

logger = logging.getLogger(__name__)

class HQQQuantizer(BaseQuantizer):
    """HQQ量化器 - Half-Quadratic Quantization，vLLM兼容"""
    
    def __init__(self):
        self.name = "hqq"
    
    def quantize(
        self,
        model,
        bits: int = 4,
        group_size: int = 128,
        layer_wise: bool = False,
        target_layers: Optional[List[int]] = None,
        calibration_data: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """执行HQQ量化"""
        logger.info(f"开始HQQ量化: {bits}bit, group_size={group_size}")
        
        # 实现HQQ量化逻辑
        # ...
        
        # 添加vLLM兼容性
        self._add_vllm_compatibility(model, bits, group_size)
        
        return model
    
    def _add_vllm_compatibility(self, model, bits: int, group_size: int):
        """为模型添加vLLM兼容性"""
        logger.info("添加vLLM兼容性...")
        
        # 添加vLLM需要的配置属性
        if hasattr(model, 'config'):
            model.config.quantization_config = {
                "quantization_method": "hqq",
                "bits": bits,
                "group_size": group_size,
                # HQQ特定参数
            }
        
        # 为所有量化层添加vLLM兼容属性
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                # 确保量化参数存在
                if not hasattr(module, 'qweight'):
                    module.qweight = module.weight.data
                if not hasattr(module, 'scales'):
                    module.scales = getattr(module, 'scale', torch.tensor(1.0))
                if not hasattr(module, 'zeros'):
                    module.zeros = getattr(module, 'zero_point', torch.tensor(0.0))
                
                # 添加vLLM需要的其他属性
                module.bits = getattr(module, 'bits', bits)
                module.group_size = group_size
                module.quantization_method = "hqq"
    
    def get_supported_bits(self) -> List[int]:
        """获取支持的量化位数"""
        return [2, 3, 4, 8]
    
    def get_supported_group_sizes(self) -> List[int]:
        """获取支持的组大小"""
        return [32, 64, 128, 256]
```

### 2. 注册量化器

在 `src/registry_init.py` 中添加注册代码：

```python
# 导入新的量化器
from .quantizers.hqq_quantizer import HQQQuantizer

def register_quantizers():
    """注册量化器"""
    logger.info("注册量化器...")
    
    # 注册GPTQ量化器
    QuantizerRegistry.register("gptq", GPTQQuantizer)
    
    # 注册AWQ量化器
    QuantizerRegistry.register("awq", AWQQuantizer)
    
    # 注册MinMax量化器
    QuantizerRegistry.register("minmax", MinMaxQuantizer)
    
    # 注册HQQ量化器
    QuantizerRegistry.register("hqq", HQQQuantizer)
```

### 3. 创建配置文件

在 `configs/` 目录下创建对应的配置文件 `hqq_config.yaml`：

```yaml
# HQQ量化配置文件
quantization:
  method: "hqq"
  bits: 4
  group_size: 128
  layer_wise: false
  outlier_suppression: null
  
  # HQQ特定参数
  hqq:
    # 添加HQQ特定的配置参数
    pass

model:
  trust_remote_code: true
  device_map: "auto"
  torch_dtype: "float16"

calibration:
  dataset: null
  num_samples: 100
  batch_size: 1
  max_length: 2048

output:
  save_format: "safetensors"
  save_quantization_config: true
```

## MinMax量化示例

MinMax量化是最基础的量化方法，这里展示如何实现：

### 1. MinMax量化器特点

- **对称量化**: 支持对称和非对称量化
- **逐通道量化**: 支持逐通道和全局量化
- **分组量化**: 支持按组进行量化
- **多种位数**: 支持2/3/4/8/16位量化
- **vLLM兼容**: 可直接在vLLM上部署

### 2. 使用MinMax量化

```bash
# 基本MinMax量化
python quantize.py --model-path /path/to/model --output-path /path/to/output --method minmax --bits 8

# 对称量化
python quantize.py --model-path /path/to/model --output-path /path/to/output --method minmax --bits 8 --config configs/minmax_config.yaml
```

### 3. MinMax配置示例

```yaml
quantization:
  method: "minmax"
  bits: 8
  group_size: 0  # 0表示不使用分组
  
  minmax:
    symmetric: false      # 是否使用对称量化
    per_channel: false    # 是否使用逐通道量化
    dynamic_range: "minmax"  # 动态范围计算方式
```

### 4. vLLM部署MinMax量化模型

```bash
# 启动vLLM服务
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/minmax_quantized/model \
    --quantization minmax

# 使用Python API
from vllm import LLM, SamplingParams

llm = LLM(model="/path/to/minmax_quantized/model", quantization="minmax")
sampling_params = SamplingParams(temperature=0.7, top_p=0.95)
outputs = llm.generate("Hello, how are you?", sampling_params)
print(outputs[0].outputs[0].text)
```

## 添加新的模型支持

### 1. 创建模型适配器

在 `src/models/` 目录下创建新的适配器文件，例如 `deepseek_adapter.py`：

```python
"""
DeepSeek模型适配器 - vLLM兼容版本
"""

import logging
from typing import Any, Dict, List
from pathlib import Path

from .base_adapter import BaseHuggingFaceAdapter

logger = logging.getLogger(__name__)

class DeepSeekAdapter(BaseHuggingFaceAdapter):
    """DeepSeek模型专用适配器，vLLM兼容"""
    
    def __init__(self):
        super().__init__()
        self.model_type = "deepseek"
    
    def get_layers(self, model):
        """获取DeepSeek模型的所有层"""
        layers = []
        
        # 根据DeepSeek的模型结构实现层获取逻辑
        # ...
        
        return layers
    
    def get_layer_by_name(self, model, layer_name: str):
        """根据名称获取DeepSeek模型的层"""
        # 实现层名称解析逻辑
        # ...
        
        return super().get_layer_by_name(model, layer_name)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取DeepSeek模型信息"""
        info = super().get_model_info()
        info["model_type"] = "deepseek"
        
        # 添加DeepSeek特有的信息
        # ...
        
        return info
    
    def _save_vllm_config(self, model, output_path: Path):
        """保存vLLM兼容的配置文件"""
        try:
            # 获取量化配置
            quantization_config = getattr(model.config, 'quantization_config', {})
            
            # 创建vLLM配置文件
            vllm_config = {
                "model": {
                    "type": "deepseek",  # DeepSeek特定类型
                    "trust_remote_code": True,
                    "dtype": "float16",
                    "quantization": quantization_config
                },
                "tokenizer": {
                    "type": "deepseek",  # DeepSeek特定类型
                    "trust_remote_code": True
                },
                "quantization": quantization_config
            }
            
            # 保存vLLM配置文件
            import json
            vllm_config_path = output_path / "vllm_config.json"
            with open(vllm_config_path, 'w', encoding='utf-8') as f:
                json.dump(vllm_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"vLLM配置文件已保存到: {vllm_config_path}")
            
        except Exception as e:
            logger.warning(f"保存vLLM配置文件失败: {e}")
```

### 2. 注册模型适配器

在 `src/registry_init.py` 中添加注册代码：

```python
# 导入新的模型适配器
from .models.deepseek_adapter import DeepSeekAdapter

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
    
    # 注册DeepSeek适配器
    ModelRegistry.register(
        "deepseek", 
        DeepSeekAdapter, 
        patterns=["deepseek", "DeepSeek", "deepseek-ai"]
    )
```

## 添加新的离群值抑制算法

### 1. 创建离群值抑制器

在 `src/outlier_suppression/` 目录下创建新的抑制器文件，例如 `outlier_suppression_v2.py`：

```python
"""
新的离群值抑制器实现 - vLLM兼容版本
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from ..core.outlier_suppression_registry import BaseOutlierSuppressor

logger = logging.getLogger(__name__)

class OutlierSuppressionV2(BaseOutlierSuppressor):
    """新的离群值抑制器，vLLM兼容"""
    
    def __init__(self):
        self.name = "outlier_suppression_v2"
    
    def apply(self, model, calibration_data: Optional[Any] = None):
        """应用离群值抑制"""
        logger.info("应用新的离群值抑制算法...")
        
        # 实现新的抑制算法
        # ...
        
        # 确保vLLM兼容性
        self._ensure_vllm_compatibility(model)
        
        logger.info("新的离群值抑制算法应用完成")
        return model
    
    def _ensure_vllm_compatibility(self, model):
        """确保vLLM兼容性"""
        # 确保所有量化层都有必要的属性
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if not hasattr(module, 'qweight'):
                    module.qweight = module.weight.data
                if not hasattr(module, 'scales'):
                    module.scales = getattr(module, 'scale', torch.tensor(1.0))
                if not hasattr(module, 'zeros'):
                    module.zeros = getattr(module, 'zero_point', torch.tensor(0.0))
    
    def get_supported_models(self) -> list:
        """获取支持的模型类型"""
        return ["llama", "qwen", "deepseek", "general"]
```

### 2. 注册离群值抑制器

在 `src/registry_init.py` 中添加注册代码：

```python
# 导入新的离群值抑制器
from .outlier_suppression.outlier_suppression_v2 import OutlierSuppressionV2

def register_outlier_suppressors():
    """注册离群值抑制器"""
    logger.info("注册离群值抑制器...")
    
    # 注册SmoothQuant抑制器
    OutlierSuppressionRegistry.register("smooth_quant", SmoothQuantSuppressor)
    
    # 注册新的抑制器
    OutlierSuppressionRegistry.register("outlier_suppression_v2", OutlierSuppressionV2)
```

## 测试新组件

### 1. 运行测试脚本

```bash
python test_framework.py
```

### 2. 验证注册

检查新组件是否正确注册：

```python
from src.core.quantizer_registry import QuantizerRegistry
from src.core.model_registry import ModelRegistry
from src.core.outlier_suppression_registry import OutlierSuppressionRegistry

# 检查量化器
print("量化器:", QuantizerRegistry.list_quantizers())

# 检查模型适配器
print("模型适配器:", ModelRegistry.list_adapters())

# 检查离群值抑制器
print("离群值抑制器:", OutlierSuppressionRegistry.list_suppressors())
```

### 3. 测试vLLM兼容性

```python
# 量化模型
python quantize.py --model-path /path/to/model --output-path /path/to/output --method your_method

# 检查生成的配置文件
import json
with open("/path/to/output/vllm_config.json", "r") as f:
    config = json.load(f)
print("vLLM配置:", config)

# 测试vLLM加载
from vllm import LLM
llm = LLM(model="/path/to/output", quantization="your_method")
print("vLLM加载成功!")
```

## 最佳实践

### 1. 错误处理

确保你的实现包含适当的错误处理：

```python
def quantize(self, model, bits: int = 4, ...):
    try:
        # 量化逻辑
        return quantized_model
    except Exception as e:
        logger.error(f"量化失败: {e}")
        raise
```

### 2. 日志记录

使用适当的日志级别记录信息：

```python
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
```

### 3. 配置验证

验证输入参数的有效性：

```python
def quantize(self, model, bits: int = 4, ...):
    if bits not in self.get_supported_bits():
        raise ValueError(f"不支持的量化位数: {bits}")
    
    if group_size not in self.get_supported_group_sizes():
        raise ValueError(f"不支持的组大小: {group_size}")
```

### 4. vLLM兼容性检查

确保所有量化层都有必要的属性：

```python
def _verify_vllm_compatibility(self, model):
    """验证vLLM兼容性"""
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            required_attrs = ['qweight', 'scales', 'zeros', 'bits', 'quantization_method']
            for attr in required_attrs:
                if not hasattr(module, attr):
                    logger.warning(f"模块 {name} 缺少vLLM兼容属性: {attr}")
```

### 5. 文档和注释

为你的代码添加详细的文档和注释：

```python
def complex_algorithm(self, input_data):
    """
    复杂的算法实现
    
    Args:
        input_data: 输入数据
        
    Returns:
        处理后的数据
        
    Raises:
        ValueError: 当输入数据无效时
    """
    # 实现逻辑
    pass
```

## 贡献指南

1. **代码风格**: 遵循PEP 8代码风格
2. **类型注解**: 使用类型注解提高代码可读性
3. **测试**: 为你的新功能编写测试
4. **文档**: 更新相关文档
5. **向后兼容**: 确保新功能不会破坏现有功能
6. **vLLM兼容**: 确保所有量化算法都支持vLLM部署

## 常见问题

### Q: 如何调试量化过程？

A: 使用详细日志模式：

```bash
python quantize.py --verbose --model-path /path/to/model --output-path /path/to/output --method gptq
```

### Q: 如何添加自定义的量化参数？

A: 在配置文件中添加自定义参数，然后在量化器中处理：

```yaml
quantization:
  method: "custom"
  custom_param1: "value1"
  custom_param2: "value2"
```

### Q: 如何支持新的模型格式？

A: 创建新的模型适配器，实现必要的接口方法。

### Q: MinMax量化的优势是什么？

A: MinMax量化是最基础的量化方法，具有以下优势：
- 实现简单，计算开销小
- 支持多种量化模式（对称/非对称，逐通道/全局）
- 不需要校准数据
- 适合快速原型和基准测试
- vLLM兼容，可直接部署

### Q: 如何确保vLLM兼容性？

A: 确保你的量化器实现以下要求：
- 保存 `qweight`, `scales`, `zeros` 等必要属性
- 在模型配置中添加 `quantization_config`
- 为每个量化层添加量化方法、位数、组大小等信息
- 生成vLLM兼容的配置文件

通过遵循这个指南，你可以轻松地扩展量化工具的功能，添加新的算法和模型支持，并确保vLLM兼容性。 