# 文档索引

欢迎使用大模型量化工具！本文档索引将帮助你快速找到所需的信息。

## 📚 文档分类

### 🚀 快速开始
- **[快速上手指南](quick_start.md)** - 5分钟快速开始，包含基本使用和常见问题
- **[4+1架构视图](4plus1_architecture.md)** - 系统架构详解，适合深入理解项目设计

### 🔧 开发指南
- **[扩展开发指南](extension_guide.md)** - 添加新的量化算法、模型支持和离群值抑制算法
- **[vLLM兼容性指南](extension_guide.md#vllm兼容性)** - 确保量化模型能在vLLM上部署
- **[设计理念与架构说明](design_philosophy.md)** - 深入理解项目设计理念和架构决策

### 📊 图表文档
- **[Mermaid图表集合](mermaid_diagrams.md)** - 系统架构、流程图、类图等可视化图表

### 📖 参考文档
- **[配置参考](../configs/)** - 各种量化方法的配置文件示例
- **[API参考](../src/)** - 源代码和API文档

## 🎯 按使用场景选择文档

### 我是新用户，想要快速上手
1. 阅读 **[快速上手指南](quick_start.md)** 的"5分钟快速开始"部分
2. 按照示例命令进行基本量化
3. 查看"常见问题"部分解决遇到的问题

### 我想要理解项目架构
1. 阅读 **[4+1架构视图](4plus1_architecture.md)** 的逻辑视图和进程视图
2. 查看 **[Mermaid图表集合](mermaid_diagrams.md)** 的系统架构图
3. 深入阅读 **[设计理念与架构说明](design_philosophy.md)** 了解设计决策
4. 查看开发视图了解代码组织
5. 阅读场景视图了解典型使用流程

### 我想要添加新的量化算法
1. 阅读 **[扩展开发指南](extension_guide.md)** 的"添加新的量化算法"部分
2. 查看 **[4+1架构视图](4plus1_architecture.md)** 的扩展点设计
3. 参考 **[Mermaid图表集合](mermaid_diagrams.md)** 的类图了解继承关系
4. 参考现有的量化器实现（如 `MinMaxQuantizer`）

### 我想要添加新的模型支持
1. 阅读 **[扩展开发指南](extension_guide.md)** 的"添加新的模型支持"部分
2. 查看 **[4+1架构视图](4plus1_architecture.md)** 的适配器层设计
3. 参考 **[Mermaid图表集合](mermaid_diagrams.md)** 的类图了解适配器结构
4. 参考现有的模型适配器实现（如 `LlamaAdapter`）

### 我想要部署到vLLM
1. 阅读 **[快速上手指南](quick_start.md)** 的"vLLM部署"部分
2. 查看 **[Mermaid图表集合](mermaid_diagrams.md)** 的部署架构图
3. 使用 `deploy_vllm.py` 工具自动部署
4. 查看vLLM兼容性要求

### 我遇到了问题
1. 查看 **[快速上手指南](quick_start.md)** 的"常见问题"部分
2. 检查 **[扩展开发指南](extension_guide.md)** 的"最佳实践"部分
3. 运行测试脚本验证环境：`python simple_test.py`

## 📋 文档结构

```
docs/
├── README.md                    # 本文档索引
├── quick_start.md              # 快速上手指南
├── 4plus1_architecture.md      # 4+1架构视图
├── design_philosophy.md        # 设计理念与架构说明
├── extension_guide.md          # 扩展开发指南
├── mermaid_diagrams.md         # Mermaid图表集合
└── mermaid_examples.md         # Mermaid使用示例
```

## 🔗 相关资源

- **[项目主页](../README.md)** - 项目概述和特性
- **[配置示例](../configs/)** - 各种量化方法的配置文件
- **[测试脚本](../tests/)** - 功能测试和验证
- **[源代码](../src/)** - 完整的源代码实现

## 💡 使用建议

1. **新手用户**：从快速上手指南开始，按照5分钟教程操作
2. **开发者**：先阅读4+1架构视图，再查看扩展开发指南
3. **运维人员**：重点关注vLLM部署和配置部分
4. **研究人员**：查看扩展开发指南，了解如何添加新算法
5. **视觉学习者**：查看Mermaid图表集合，直观理解系统架构

## 🐛 反馈和建议

如果你发现文档中的问题或有改进建议，请：
1. 检查是否在常见问题中已有解答
2. 查看源代码中的注释和文档字符串
3. 参考扩展开发指南中的最佳实践

---

**提示**: 建议先阅读快速上手指南，然后根据具体需求选择相应的文档深入阅读。对于视觉学习者，Mermaid图表集合提供了直观的系统理解方式。 