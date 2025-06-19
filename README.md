# 大模型量化工具 (LLM Quantization Toolkit)

一个灵活、可扩展的大模型量化工具，支持多种量化算法和模型架构，**可直接在vLLM上部署**。

## 📖 文档导航

- **[🚀 快速上手指南](docs/quick_start.md)** - 5分钟快速开始
- **[🏗️ 4+1架构视图](docs/4plus1_architecture.md)** - 系统架构详解
- **[🎯 设计理念说明](docs/design_philosophy.md)** - 深入理解设计决策
- **[📊 Mermaid图表](docs/mermaid_diagrams.md)** - 可视化架构图
- **[🔧 扩展开发指南](docs/extension_guide.md)** - 添加新算法和模型

## 特性

- 🚀 **多模型支持**: Llama, Qwen, DeepSeek, 等主流大模型
- 🔧 **多量化算法**: GPTQ, AWQ, MinMax, HQQ, 等先进量化方法
- 🎯 **离群值抑制**: SmoothQuant, 等算法支持
- 📊 **逐层量化**: 支持按层进行量化
- 🔌 **可扩展架构**: 易于添加新的量化算法和模型支持
- 🖥️ **命令行工具**: 简单易用的CLI接口
- ⚡ **vLLM兼容**: 量化的模型可直接在vLLM上部署

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 基本量化

```bash
python quantize.py --model-path /path/to/model --output-path /path/to/output --method gptq
```

### MinMax量化

```bash
python quantize.py --model-path /path/to/model --output-path /path/to/output --method minmax --bits 8
```

### 逐层量化

```bash
python quantize.py --model-path /path/to/model --output-path /path/to/output --method awq --layer-wise
```

### 使用离群值抑制

```bash
python quantize.py --model-path /path/to/model --output-path /path/to/output --method gptq --outlier-suppression smooth_quant
```

## vLLM部署

量化的模型可以直接在vLLM上部署，无需额外转换：

### 1. 安装vLLM

```bash
pip install vllm
```

### 2. 启动vLLM服务

```bash
# GPTQ量化模型
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/quantized/model \
    --quantization gptq

# AWQ量化模型
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/quantized/model \
    --quantization awq

# MinMax量化模型
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/quantized/model \
    --quantization minmax
```

### 3. 使用API

```python
from vllm import LLM, SamplingParams

# 加载量化模型
llm = LLM(model="/path/to/quantized/model", quantization="gptq")

# 生成文本
sampling_params = SamplingParams(temperature=0.7, top_p=0.95)
outputs = llm.generate("Hello, how are you?", sampling_params)
print(outputs[0].outputs[0].text)
```

## 支持的模型

- Llama 系列 (Llama-2, Llama-3)
- Qwen 系列 (Qwen-1.5, Qwen-2)
- DeepSeek 系列
- 其他兼容 HuggingFace Transformers 的模型

## 支持的量化算法

- **GPTQ**: 基于Hessian矩阵的量化方法，vLLM原生支持
- **AWQ**: Activation-aware Weight Quantization，vLLM原生支持
- **MinMax**: 基础的线性量化方法，支持对称/非对称量化，vLLM兼容
- **HQQ**: Half-Quadratic Quantization
- **更多算法可通过插件扩展**

## 支持的离群值抑制算法

- **SmoothQuant**: 平滑量化算法
- **更多算法可通过插件扩展**

## 配置示例

查看 `configs/` 目录下的配置文件示例。

## 扩展开发

### 添加新的量化算法

1. 在 `quantizers/` 目录下创建新的量化器类
2. 继承 `BaseQuantizer` 类
3. 实现 `quantize()` 方法
4. 添加vLLM兼容性支持
5. 在 `quantizer_registry.py` 中注册

### 添加新的模型支持

1. 在 `models/` 目录下创建新的模型适配器
2. 继承 `BaseModelAdapter` 类
3. 实现必要的方法
4. 在 `model_registry.py` 中注册

## 许可证

MIT License 