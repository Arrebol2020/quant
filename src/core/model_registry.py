"""
模型注册表 - 管理不同模型的适配器
"""

import logging
from typing import Dict, Type, Optional
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseModelAdapter(ABC):
    """模型适配器基类"""
    
    @abstractmethod
    def load_model(self, model_path: Path):
        """加载模型"""
        pass
    
    @abstractmethod
    def save_model(self, model, output_path: Path):
        """保存模型"""
        pass
    
    @abstractmethod
    def load_calibration_data(self, dataset_path: str):
        """加载校准数据"""
        pass
    
    @abstractmethod
    def get_layers(self, model):
        """获取模型的所有层"""
        pass
    
    @abstractmethod
    def get_layer_by_name(self, model, layer_name: str):
        """根据名称获取层"""
        pass

class ModelRegistry:
    """模型注册表"""
    
    _adapters: Dict[str, Type[BaseModelAdapter]] = {}
    _model_patterns: Dict[str, str] = {}
    
    @classmethod
    def register(cls, name: str, adapter_class: Type[BaseModelAdapter], patterns: list = None):
        """注册模型适配器"""
        cls._adapters[name] = adapter_class
        if patterns:
            for pattern in patterns:
                cls._model_patterns[pattern] = name
        logger.info(f"注册模型适配器: {name}")
    
    @classmethod
    def get_adapter(cls, model_path: Path) -> BaseModelAdapter:
        """根据模型路径获取适配器"""
        # 尝试通过路径模式匹配
        model_path_str = str(model_path)
        for pattern, adapter_name in cls._model_patterns.items():
            if pattern in model_path_str:
                adapter_class = cls._adapters[adapter_name]
                return adapter_class()
        
        # 如果没有匹配到，使用默认适配器
        if "default" in cls._adapters:
            return cls._adapters["default"]()
        
        raise ValueError(f"未找到适合模型 {model_path} 的适配器")
    
    @classmethod
    def list_adapters(cls) -> list:
        """列出所有可用的适配器"""
        return list(cls._adapters.keys()) 