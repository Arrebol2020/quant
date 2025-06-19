"""
基础模型适配器 - 提供通用的模型操作功能，vLLM兼容版本
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from ..core.model_registry import BaseModelAdapter

logger = logging.getLogger(__name__)

class BaseHuggingFaceAdapter(BaseModelAdapter):
    """基于HuggingFace的基础模型适配器，vLLM兼容"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
    
    def load_model(self, model_path: Path):
        """加载模型"""
        logger.info(f"加载模型: {model_path}")
        
        try:
            # 加载配置
            self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                use_fast=False
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=self.config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            logger.info(f"模型加载成功: {self.model.__class__.__name__}")
            return self.model
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def save_model(self, model, output_path: Path):
        """保存模型（vLLM兼容格式）"""
        logger.info(f"保存模型到: {output_path}")
        
        try:
            # 确保输出目录存在
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存模型
            model.save_pretrained(output_path)
            
            # 保存分词器
            if self.tokenizer:
                self.tokenizer.save_pretrained(output_path)
            
            # 保存配置
            if self.config:
                self.config.save_pretrained(output_path)
            
            # 保存vLLM兼容的配置文件
            self._save_vllm_config(model, output_path)
            
            logger.info("模型保存成功，已添加vLLM兼容配置")
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            raise
    
    def _save_vllm_config(self, model, output_path: Path):
        """保存vLLM兼容的配置文件"""
        try:
            # 获取量化配置
            quantization_config = getattr(model.config, 'quantization_config', {})
            
            # 创建vLLM配置文件
            vllm_config = {
                "model": {
                    "type": "llama",  # 默认类型，可根据实际模型调整
                    "trust_remote_code": True,
                    "dtype": "float16",
                    "quantization": quantization_config
                },
                "tokenizer": {
                    "type": "llama",  # 默认类型，可根据实际模型调整
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
    
    def load_calibration_data(self, dataset_path: str):
        """加载校准数据"""
        logger.info(f"加载校准数据: {dataset_path}")
        
        try:
            from datasets import load_dataset
            
            # 加载数据集
            dataset = load_dataset(dataset_path)
            
            # 预处理数据
            if "train" in dataset:
                calibration_data = dataset["train"]
            elif "validation" in dataset:
                calibration_data = dataset["validation"]
            else:
                calibration_data = dataset[list(dataset.keys())[0]]
            
            logger.info(f"校准数据加载成功: {len(calibration_data)} 样本")
            return calibration_data
            
        except Exception as e:
            logger.error(f"校准数据加载失败: {e}")
            raise
    
    def get_layers(self, model):
        """获取模型的所有层"""
        layers = []
        
        # 遍历模型的所有模块
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                layers.append((name, module))
        
        return layers
    
    def get_layer_by_name(self, model, layer_name: str):
        """根据名称获取层"""
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.model:
            return {}
        
        info = {
            "model_type": self.model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        
        if self.config:
            info.update({
                "hidden_size": getattr(self.config, 'hidden_size', None),
                "num_attention_heads": getattr(self.config, 'num_attention_heads', None),
                "num_hidden_layers": getattr(self.config, 'num_hidden_layers', None),
                "vocab_size": getattr(self.config, 'vocab_size', None),
            })
        
        # 添加量化信息
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'quantization_config'):
            info["quantization"] = self.model.config.quantization_config
        
        return info 