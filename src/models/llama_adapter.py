"""
Llama模型适配器
"""

import logging
from typing import Any, Dict, List
from pathlib import Path

from .base_adapter import BaseHuggingFaceAdapter

logger = logging.getLogger(__name__)

class LlamaAdapter(BaseHuggingFaceAdapter):
    """Llama模型专用适配器"""
    
    def __init__(self):
        super().__init__()
        self.model_type = "llama"
    
    def get_layers(self, model):
        """获取Llama模型的所有层"""
        layers = []
        
        # Llama模型的层结构
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for i, layer in enumerate(model.model.layers):
                # 获取注意力层
                if hasattr(layer, 'self_attn'):
                    layers.append((f"model.layers.{i}.self_attn", layer.self_attn))
                
                # 获取MLP层
                if hasattr(layer, 'mlp'):
                    layers.append((f"model.layers.{i}.mlp", layer.mlp))
        
        return layers
    
    def get_layer_by_name(self, model, layer_name: str):
        """根据名称获取Llama模型的层"""
        # 解析层名称
        parts = layer_name.split('.')
        
        if len(parts) >= 4 and parts[0] == 'model' and parts[1] == 'layers':
            layer_idx = int(parts[2])
            layer_type = parts[3]
            
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layer = model.model.layers[layer_idx]
                
                if layer_type == 'self_attn':
                    return layer.self_attn
                elif layer_type == 'mlp':
                    return layer.mlp
        
        return super().get_layer_by_name(model, layer_name)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取Llama模型信息"""
        info = super().get_model_info()
        info["model_type"] = "llama"
        
        if self.config:
            info.update({
                "rope_theta": getattr(self.config, 'rope_theta', None),
                "max_position_embeddings": getattr(self.config, 'max_position_embeddings', None),
            })
        
        return info 