"""
离群值抑制注册表 - 管理不同的离群值抑制算法
"""

import logging
from typing import Dict, Type, Optional, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseOutlierSuppressor(ABC):
    """离群值抑制器基类"""
    
    @abstractmethod
    def apply(self, model, calibration_data: Optional[Any] = None):
        """应用离群值抑制"""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> list:
        """获取支持的模型类型"""
        pass

class OutlierSuppressionRegistry:
    """离群值抑制注册表"""
    
    _suppressors: Dict[str, Type[BaseOutlierSuppressor]] = {}
    
    @classmethod
    def register(cls, name: str, suppressor_class: Type[BaseOutlierSuppressor]):
        """注册离群值抑制器"""
        cls._suppressors[name] = suppressor_class
        logger.info(f"注册离群值抑制器: {name}")
    
    @classmethod
    def get_suppressor(cls, name: str) -> BaseOutlierSuppressor:
        """获取离群值抑制器实例"""
        if name not in cls._suppressors:
            raise ValueError(f"未找到离群值抑制器: {name}")
        return cls._suppressors[name]()
    
    @classmethod
    def list_suppressors(cls) -> list:
        """列出所有可用的离群值抑制器"""
        return list(cls._suppressors.keys())
    
    @classmethod
    def get_suppressor_info(cls, name: str) -> Dict[str, Any]:
        """获取离群值抑制器信息"""
        if name not in cls._suppressors:
            raise ValueError(f"未找到离群值抑制器: {name}")
        
        suppressor = cls._suppressors[name]()
        return {
            "name": name,
            "supported_models": suppressor.get_supported_models()
        } 