"""
量化器注册表 - 管理不同的量化算法
"""

import logging
from typing import Dict, Type, Optional, List, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseQuantizer(ABC):
    """量化器基类"""
    
    @abstractmethod
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
        """执行量化"""
        pass
    
    @abstractmethod
    def get_supported_bits(self) -> List[int]:
        """获取支持的量化位数"""
        pass
    
    @abstractmethod
    def get_supported_group_sizes(self) -> List[int]:
        """获取支持的组大小"""
        pass

class QuantizerRegistry:
    """量化器注册表"""
    
    _quantizers: Dict[str, Type[BaseQuantizer]] = {}
    
    @classmethod
    def register(cls, name: str, quantizer_class: Type[BaseQuantizer]):
        """注册量化器"""
        cls._quantizers[name] = quantizer_class
        logger.info(f"注册量化器: {name}")
    
    @classmethod
    def get_quantizer(cls, name: str) -> BaseQuantizer:
        """获取量化器实例"""
        if name not in cls._quantizers:
            raise ValueError(f"未找到量化器: {name}")
        return cls._quantizers[name]()
    
    @classmethod
    def list_quantizers(cls) -> list:
        """列出所有可用的量化器"""
        return list(cls._quantizers.keys())
    
    @classmethod
    def get_quantizer_info(cls, name: str) -> Dict[str, Any]:
        """获取量化器信息"""
        if name not in cls._quantizers:
            raise ValueError(f"未找到量化器: {name}")
        
        quantizer = cls._quantizers[name]()
        return {
            "name": name,
            "supported_bits": quantizer.get_supported_bits(),
            "supported_group_sizes": quantizer.get_supported_group_sizes()
        } 