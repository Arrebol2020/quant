# 快速上手指南

## 🚀 5分钟快速开始

### 1. 安装环境

```bash
# 克隆项目（如果还没有）
git clone <your-repo-url>
cd quant

# 安装依赖
pip install -r requirements.txt

# 验证安装
python simple_test.py
```

### 2. 基本量化（2分钟）

```bash
# 量化一个模型
python quantize.py \
    --model-path /path/to/your/model \
    --output-path ./quantized_model \
    --method gptq \
    --bits 4
```

### 3. 部署到vLLM（1分钟）

```bash
# 使用部署工具
python deploy_vllm.py --model-path ./quantized_model

# 或者直接启动vLLM
python -m vllm.entrypoints.openai.api_server \
    --model ./quantized_model \
    --quantization gptq
```

### 4. 测试API（1分钟）

```bash
# 测试API
curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello, how are you?", "max_tokens": 50}'
```

## 📚 详细使用指南

### 支持的量化方法

| 方法 | 描述 | 适用场景 | 示例命令 |
|------|------|----------|----------|
| **GPTQ** | 基于Hessian矩阵的量化 | 高质量量化 | `--method gptq` |
| **AWQ** | Activation-aware量化 | 激活感知量化 | `--method awq` |
| **MinMax** | 基础线性量化 | 快速原型 | `--method minmax` |

### 常用参数

```bash
python quantize.py \
    --model-path /path/to/model \      # 原始模型路径
    --output-path /path/to/output \    # 输出路径
    --method gptq \                    # 量化方法
    --bits 4 \                         # 量化位数 (2/3/4/8/16)
    --group-size 128 \                 # 量化组大小
    --layer-wise \                     # 逐层量化
    --layers 0,1,2 \                   # 指定层
    --outlier-suppression smooth_quant \ # 离群值抑制
    --config configs/gptq_config.yaml \ # 配置文件
    --verbose                          # 详细输出
```

### 高级功能

#### 1. 逐层量化

```bash
# 只量化前3层
python quantize.py \
    --model-path /path/to/model \
    --output-path /path/to/output \
    --method gptq \
    --layer-wise \
    --layers 0,1,2
```

#### 2. 使用离群值抑制

```bash
# 使用SmoothQuant提高质量
python quantize.py \
    --model-path /path/to/model \
    --output-path /path/to/output \
    --method gptq \
    --outlier-suppression smooth_quant \
    --calibration-dataset /path/to/calibration_data
```

#### 3. 自定义配置

```bash
# 使用配置文件
python quantize.py \
    --model-path /path/to/model \
    --output-path /path/to/output \
    --method gptq \
    --config configs/gptq_config.yaml
```

## 🔧 配置说明

### 基本配置示例

```yaml
# configs/gptq_config.yaml
quantization:
  method: "gptq"
  bits: 4
  group_size: 128
  layer_wise: false
  outlier_suppression: null

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

### 不同量化方法的配置

#### GPTQ配置
```yaml
quantization:
  method: "gptq"
  gptq:
    damp_percent: 0.01
    desc_act: false
    static_groups: false
    sym: true
    true_sequential: true
```

#### AWQ配置
```yaml
quantization:
  method: "awq"
  awq:
    zero_point: true
    q_type: "asym"
    w_bit: 4
    a_bit: 8
```

#### MinMax配置
```yaml
quantization:
  method: "minmax"
  minmax:
    symmetric: false
    per_channel: false
    dynamic_range: "minmax"
```

## 🚀 vLLM部署

### 自动部署

```bash
# 使用部署工具（推荐）
python deploy_vllm.py --model-path ./quantized_model
```

### 手动部署

```bash
# GPTQ模型
python -m vllm.entrypoints.openai.api_server \
    --model ./quantized_model \
    --quantization gptq

# AWQ模型
python -m vllm.entrypoints.openai.api_server \
    --model ./quantized_model \
    --quantization awq

# MinMax模型
python -m vllm.entrypoints.openai.api_server \
    --model ./quantized_model \
    --quantization minmax
```

### Python API使用

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="./quantized_model",
    quantization="gptq"  # 或 "awq", "minmax"
)

# 设置参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100
)

# 生成文本
outputs = llm.generate("Hello, how are you?", sampling_params)
print(outputs[0].outputs[0].text)
```

### HTTP API使用

```bash
# 文本补全
curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Hello, how are you?",
        "max_tokens": 50,
        "temperature": 0.7
    }'

# 聊天补全
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }'
```

## 🐛 常见问题

### Q1: 内存不足怎么办？

**A**: 尝试以下方法：
- 使用更小的量化位数：`--bits 2` 或 `--bits 3`
- 使用逐层量化：`--layer-wise --layers 0,1,2`
- 减少batch size：在配置文件中设置 `batch_size: 1`

### Q2: 量化速度太慢？

**A**: 优化建议：
- 使用MinMax量化：`--method minmax`
- 减少校准数据：`num_samples: 50`
- 使用更小的组大小：`--group-size 64`

### Q3: 量化质量不好？

**A**: 提高质量的方法：
- 使用离群值抑制：`--outlier-suppression smooth_quant`
- 增加校准数据量
- 使用更高的量化位数：`--bits 8`

### Q4: vLLM部署失败？

**A**: 检查以下几点：
- 确认量化方法正确：`--quantization gptq/awq/minmax`
- 检查模型路径是否存在
- 查看vLLM配置文件：`vllm_config.json`

### Q5: 如何添加新的量化算法？

**A**: 参考扩展指南：
```bash
# 查看详细指南
cat docs/extension_guide.md

# 查看架构文档
cat docs/4plus1_architecture.md
```

## 📊 性能对比

| 量化方法 | 速度 | 质量 | 内存占用 | 适用场景 |
|----------|------|------|----------|----------|
| **GPTQ** | 中等 | 高 | 中等 | 生产环境 |
| **AWQ** | 快 | 高 | 低 | 资源受限 |
| **MinMax** | 最快 | 中等 | 最低 | 快速原型 |

## 🔗 相关链接

- [完整文档](docs/4plus1_architecture.md) - 4+1架构视图
- [扩展指南](docs/extension_guide.md) - 开发扩展
- [配置示例](configs/) - 配置文件
- [测试脚本](tests/) - 功能测试

## 💡 最佳实践

1. **选择合适的量化方法**：
   - 生产环境：GPTQ或AWQ
   - 快速测试：MinMax
   - 资源受限：AWQ

2. **优化量化参数**：
   - 从4位开始，根据需要调整
   - 使用128的组大小作为起点
   - 对于重要模型使用离群值抑制

3. **部署建议**：
   - 使用部署工具自动检测量化方法
   - 在生产环境中使用配置文件
   - 定期备份原始模型

4. **监控和调试**：
   - 使用`--verbose`查看详细日志
   - 检查量化报告了解效果
   - 对比不同量化方法的结果 