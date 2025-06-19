"""
Qwen模型适配器
"""

import logging
from typing import Any, Dict, List
from pathlib import Path

from .base_adapter import BaseHuggingFaceAdapter

logger = logging.getLogger(__name__)

class QwenAdapter(BaseHuggingFaceAdapter):
    """Qwen模型专用适配器"""
    
    def __init__(self):
        super().__init__()
        self.model_type = "qwen"
    
    def get_layers(self, model):
        """获取Qwen模型的所有层"""
        layers = []
        
        # Qwen模型的层结构
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            for i, layer in enumerate(model.transformer.h):
                # 获取注意力层
                if hasattr(layer, 'attn'):
                    layers.append((f"transformer.h.{i}.attn", layer.attn))
                
                # 获取MLP层
                if hasattr(layer, 'mlp'):
                    layers.append((f"transformer.h.{i}.mlp", layer.mlp))
        
        return layers
    
    def get_layer_by_name(self, model, layer_name: str):
        """根据名称获取Qwen模型的层"""
        # 解析层名称
        parts = layer_name.split('.')
        
        if len(parts) >= 4 and parts[0] == 'transformer' and parts[1] == 'h':
            layer_idx = int(parts[2])
            layer_type = parts[3]
            
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                layer = model.transformer.h[layer_idx]
                
                if layer_type == 'attn':
                    return layer.attn
                elif layer_type == 'mlp':
                    return layer.mlp
        
        return super().get_layer_by_name(model, layer_name)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取Qwen模型信息"""
        info = super().get_model_info()
        info["model_type"] = "qwen"
        
        if self.config:
            info.update({
                "rope_theta": getattr(self.config, 'rope_theta', None),
                "max_position_embeddings": getattr(self.config, 'max_position_embeddings', None),
                "use_flash_attention_2": getattr(self.config, 'use_flash_attention_2', False),
            })
        
        return info 