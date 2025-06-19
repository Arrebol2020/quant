# Mermaid 图表集合

本文档包含使用 Mermaid 绘制的各种图表，用于展示大模型量化工具的系统架构、流程和组件关系。

## 1. 系统架构图

### 1.1 整体架构

```mermaid
graph TB
    subgraph "用户界面层"
        CLI[命令行工具<br/>quantize.py]
        Deploy[部署工具<br/>deploy_vllm.py]
    end
    
    subgraph "核心管理层"
        QM[量化管理器<br/>QuantizationManager]
        MR[模型注册表<br/>ModelRegistry]
        QR[量化器注册表<br/>QuantizerRegistry]
        OR[离群值抑制注册表<br/>OutlierSuppressionRegistry]
    end
    
    subgraph "适配器层"
        BA[基础适配器<br/>BaseHuggingFaceAdapter]
        LA[Llama适配器<br/>LlamaAdapter]
        QA[Qwen适配器<br/>QwenAdapter]
    end
    
    subgraph "量化算法层"
        GQ[GPTQ量化器<br/>GPTQQuantizer]
        AQ[AWQ量化器<br/>AWQQuantizer]
        MQ[MinMax量化器<br/>MinMaxQuantizer]
    end
    
    subgraph "离群值抑制层"
        SQ[SmoothQuant抑制器<br/>SmoothQuantSuppressor]
    end
    
    subgraph "外部系统"
        vLLM[vLLM服务]
        HF[HuggingFace模型]
    end
    
    CLI --> QM
    Deploy --> vLLM
    QM --> MR
    QM --> QR
    QM --> OR
    MR --> BA
    MR --> LA
    MR --> QA
    QR --> GQ
    QR --> AQ
    QR --> MQ
    OR --> SQ
    BA --> HF
    LA --> HF
    QA --> HF
    GQ --> vLLM
    AQ --> vLLM
    MQ --> vLLM
```

### 1.2 组件关系图

```mermaid
graph LR
    subgraph "注册表系统"
        MR[ModelRegistry]
        QR[QuantizerRegistry]
        OR[OutlierSuppressionRegistry]
    end
    
    subgraph "量化管理器"
        QM[QuantizationManager]
    end
    
    subgraph "适配器"
        BA[BaseAdapter]
        LA[LlamaAdapter]
        QA[QwenAdapter]
    end
    
    subgraph "量化器"
        GQ[GPTQQuantizer]
        AQ[AWQQuantizer]
        MQ[MinMaxQuantizer]
    end
    
    subgraph "离群值抑制"
        SQ[SmoothQuant]
    end
    
    QM --> MR
    QM --> QR
    QM --> OR
    MR --> BA
    MR --> LA
    MR --> QA
    QR --> GQ
    QR --> AQ
    QR --> MQ
    OR --> SQ
```

## 2. 流程图

### 2.1 量化流程

```mermaid
flowchart TD
    Start([开始]) --> Parse[解析命令行参数]
    Parse --> Config[加载配置文件]
    Config --> Init[初始化量化管理器]
    Init --> LoadModel[加载模型]
    LoadModel --> SelectAdapter[选择模型适配器]
    SelectAdapter --> SelectQuantizer[选择量化器]
    SelectQuantizer --> CheckOutlier{需要离群值抑制?}
    CheckOutlier -->|是| ApplyOutlier[应用离群值抑制]
    CheckOutlier -->|否| Quantize[执行量化]
    ApplyOutlier --> Quantize
    Quantize --> AddVLLM[添加vLLM兼容性]
    AddVLLM --> Save[保存量化模型]
    Save --> Report[生成量化报告]
    Report --> End([结束])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style Quantize fill:#87CEEB
    style AddVLLM fill:#DDA0DD
```

### 2.2 逐层量化流程

```mermaid
flowchart TD
    Start([开始逐层量化]) --> LoadModel[加载模型]
    LoadModel --> GetLayers[获取所有层]
    GetLayers --> FilterLayers[过滤目标层]
    FilterLayers --> InitProgress[初始化进度条]
    InitProgress --> Loop{还有层需要量化?}
    Loop -->|是| SelectLayer[选择下一层]
    SelectLayer --> CheckLayer{层有权重?}
    CheckLayer -->|否| Loop
    CheckLayer -->|是| QuantizeLayer[量化当前层]
    QuantizeLayer --> UpdateProgress[更新进度]
    UpdateProgress --> Loop
    Loop -->|否| AddVLLM[添加vLLM兼容性]
    AddVLLM --> Save[保存模型]
    Save --> End([结束])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style QuantizeLayer fill:#87CEEB
```

### 2.3 注册表工作流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant QM as 量化管理器
    participant MR as 模型注册表
    participant QR as 量化器注册表
    participant OR as 离群值抑制注册表
    participant A as 适配器
    participant Q as 量化器
    participant O as 离群值抑制器
    
    U->>QM: 执行量化命令
    QM->>MR: 获取模型适配器
    MR->>A: 返回适配器实例
    QM->>QR: 获取量化器
    QR->>Q: 返回量化器实例
    QM->>OR: 获取离群值抑制器
    OR->>O: 返回抑制器实例
    QM->>A: 加载模型
    A->>QM: 返回模型
    QM->>O: 应用离群值抑制
    O->>QM: 返回处理后的模型
    QM->>Q: 执行量化
    Q->>QM: 返回量化模型
    QM->>A: 保存模型
    A->>QM: 保存完成
    QM->>U: 返回结果
```

## 3. 类图

### 3.1 核心类关系

```mermaid
classDiagram
    class BaseQuantizer {
        <<abstract>>
        +name: str
        +quantize(model, bits, group_size, ...)
        +get_supported_bits()
        +get_supported_group_sizes()
        +_add_vllm_compatibility()
    }
    
    class GPTQQuantizer {
        +quantize()
        +_quantize_full_model()
        +_quantize_layer_wise()
        +_add_vllm_compatibility()
    }
    
    class AWQQuantizer {
        +quantize()
        +_quantize_layer_awq()
        +_compute_awq_params()
        +_add_vllm_compatibility()
    }
    
    class MinMaxQuantizer {
        +quantize()
        +_quantize_layer_minmax()
        +_compute_global_params()
        +_add_vllm_compatibility()
    }
    
    class BaseModelAdapter {
        <<abstract>>
        +model
        +tokenizer
        +config
        +load_model(path)
        +save_model(model, path)
        +get_layers(model)
    }
    
    class BaseHuggingFaceAdapter {
        +load_model()
        +save_model()
        +_save_vllm_config()
        +get_model_info()
    }
    
    class LlamaAdapter {
        +get_layers()
        +get_model_info()
    }
    
    class QwenAdapter {
        +get_layers()
        +get_model_info()
    }
    
    BaseQuantizer <|-- GPTQQuantizer
    BaseQuantizer <|-- AWQQuantizer
    BaseQuantizer <|-- MinMaxQuantizer
    BaseModelAdapter <|-- BaseHuggingFaceAdapter
    BaseHuggingFaceAdapter <|-- LlamaAdapter
    BaseHuggingFaceAdapter <|-- QwenAdapter
```

### 3.2 注册表类关系

```mermaid
classDiagram
    class Registry {
        <<abstract>>
        +_registry: Dict
        +register(name, component, patterns)
        +get(name)
        +list_components()
    }
    
    class ModelRegistry {
        +register(name, adapter, patterns)
        +get_adapter(model_path)
        +list_adapters()
    }
    
    class QuantizerRegistry {
        +register(name, quantizer)
        +get_quantizer(method)
        +list_quantizers()
    }
    
    class OutlierSuppressionRegistry {
        +register(name, suppressor)
        +get_suppressor(method)
        +list_suppressors()
    }
    
    Registry <|-- ModelRegistry
    Registry <|-- QuantizerRegistry
    Registry <|-- OutlierSuppressionRegistry
```

## 4. 部署架构图

### 4.1 vLLM部署流程

```mermaid
graph TB
    subgraph "量化阶段"
        Model[原始模型]
        Quantize[量化工具]
        QuantizedModel[量化模型]
        Model --> Quantize
        Quantize --> QuantizedModel
    end
    
    subgraph "部署阶段"
        vLLM[vLLM服务]
        API[API接口]
        Client[客户端]
        QuantizedModel --> vLLM
        vLLM --> API
        API --> Client
    end
    
    subgraph "存储"
        Config[vLLM配置文件]
        Weights[模型权重]
        Tokenizer[分词器]
        QuantizedModel --> Config
        QuantizedModel --> Weights
        QuantizedModel --> Tokenizer
    end
```

### 4.2 文件结构图

```mermaid
graph TD
    subgraph "项目根目录"
        Root[quant/]
    end
    
    subgraph "源代码"
        Src[src/]
        Core[core/]
        Models[models/]
        Quantizers[quantizers/]
        Outlier[outlier_suppression/]
        Utils[utils/]
    end
    
    subgraph "配置文件"
        Configs[configs/]
        GPTQConfig[gptq_config.yaml]
        AWQConfig[awq_config.yaml]
        MinMaxConfig[minmax_config.yaml]
    end
    
    subgraph "文档"
        Docs[docs/]
        QuickStart[quick_start.md]
        Architecture[4plus1_architecture.md]
        Extension[extension_guide.md]
    end
    
    subgraph "工具脚本"
        Quantize[quantize.py]
        Deploy[deploy_vllm.py]
        Test[simple_test.py]
    end
    
    Root --> Src
    Root --> Configs
    Root --> Docs
    Root --> Quantize
    Root --> Deploy
    Root --> Test
    
    Src --> Core
    Src --> Models
    Src --> Quantizers
    Src --> Outlier
    Src --> Utils
    
    Configs --> GPTQConfig
    Configs --> AWQConfig
    Configs --> MinMaxConfig
    
    Docs --> QuickStart
    Docs --> Architecture
    Docs --> Extension
```

## 5. 数据流图

### 5.1 量化数据流

```mermaid
flowchart LR
    subgraph "输入"
        OriginalModel[原始模型]
        Config[配置文件]
        CalibrationData[校准数据]
    end
    
    subgraph "处理"
        Adapter[模型适配器]
        Quantizer[量化器]
        OutlierSuppressor[离群值抑制器]
    end
    
    subgraph "输出"
        QuantizedModel[量化模型]
        VLLMConfig[vLLM配置]
        Report[量化报告]
    end
    
    OriginalModel --> Adapter
    Config --> Adapter
    Config --> Quantizer
    CalibrationData --> OutlierSuppressor
    
    Adapter --> Quantizer
    OutlierSuppressor --> Quantizer
    Quantizer --> QuantizedModel
    Quantizer --> VLLMConfig
    Quantizer --> Report
```

### 5.2 配置数据流

```mermaid
flowchart TD
    subgraph "配置源"
        CLI[命令行参数]
        YAML[YAML配置文件]
        Default[默认配置]
    end
    
    subgraph "配置处理"
        Parser[配置解析器]
        Validator[配置验证器]
        Merger[配置合并器]
    end
    
    subgraph "配置使用"
        Quantization[量化配置]
        Model[模型配置]
        Calibration[校准配置]
        Output[输出配置]
    end
    
    CLI --> Parser
    YAML --> Parser
    Default --> Parser
    
    Parser --> Validator
    Validator --> Merger
    Merger --> Quantization
    Merger --> Model
    Merger --> Calibration
    Merger --> Output
```

## 6. 状态图

### 6.1 量化状态转换

```mermaid
stateDiagram-v2
    [*] --> Idle: 初始化
    Idle --> Loading: 开始加载模型
    Loading --> Loaded: 模型加载成功
    Loading --> Error: 加载失败
    Loaded --> Quantizing: 开始量化
    Quantizing --> Quantized: 量化完成
    Quantizing --> Error: 量化失败
    Quantized --> Saving: 开始保存
    Saving --> Saved: 保存完成
    Saving --> Error: 保存失败
    Saved --> [*]: 结束
    Error --> [*]: 错误处理
```

### 6.2 注册表状态

```mermaid
stateDiagram-v2
    [*] --> Empty: 创建注册表
    Empty --> Registered: 注册组件
    Registered --> Registered: 继续注册
    Registered --> Retrieved: 获取组件
    Retrieved --> Registered: 返回注册状态
    Registered --> [*]: 销毁注册表
```

## 7. 甘特图

### 7.1 量化任务时间线

```mermaid
gantt
    title 量化任务时间线
    dateFormat  YYYY-MM-DD
    section 准备阶段
    环境检查           :done, env, 2024-01-01, 1d
    模型下载           :done, download, 2024-01-02, 2d
    配置准备           :done, config, 2024-01-03, 1d
    
    section 量化阶段
    模型加载           :active, load, 2024-01-04, 1d
    量化执行           :quantize, 2024-01-05, 3d
    结果保存           :save, 2024-01-08, 1d
    
    section 部署阶段
    vLLM部署           :deploy, 2024-01-09, 1d
    测试验证           :test, 2024-01-10, 1d
```

## 8. 饼图

### 8.1 量化方法分布

```mermaid
pie title 量化方法使用分布
    "GPTQ" : 45
    "AWQ" : 30
    "MinMax" : 15
    "其他" : 10
```

### 8.2 模型类型分布

```mermaid
pie title 支持的模型类型
    "Llama系列" : 40
    "Qwen系列" : 25
    "DeepSeek系列" : 20
    "其他模型" : 15
```

## 使用说明

这些Mermaid图表可以：

1. **在Markdown中使用**：直接复制图表代码到Markdown文件中
2. **在GitHub中显示**：GitHub原生支持Mermaid图表
3. **在文档工具中渲染**：如GitBook、Docusaurus等
4. **在线编辑**：使用 [Mermaid Live Editor](https://mermaid.live/)

## 图表维护

- 当系统架构发生变化时，请及时更新相应的图表
- 保持图表简洁明了，避免过于复杂
- 使用统一的颜色和样式规范
- 为图表添加适当的标题和说明
``` 