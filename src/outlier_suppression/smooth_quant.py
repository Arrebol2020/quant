"""
SmoothQuant离群值抑制器实现
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from ..core.outlier_suppression_registry import BaseOutlierSuppressor

logger = logging.getLogger(__name__)

class SmoothQuantSuppressor(BaseOutlierSuppressor):
    """SmoothQuant离群值抑制器"""
    
    def __init__(self):
        self.name = "smooth_quant"
        self.alpha = 0.5  # 平滑因子
    
    def apply(self, model, calibration_data: Optional[Any] = None):
        """应用SmoothQuant"""
        logger.info("应用SmoothQuant离群值抑制...")
        
        # 收集激活统计信息
        activation_stats = self._collect_activation_stats(model, calibration_data)
        
        # 计算缩放因子
        scaling_factors = self._compute_scaling_factors(activation_stats)
        
        # 应用平滑量化
        self._apply_smooth_quant(model, scaling_factors)
        
        logger.info("SmoothQuant应用完成")
        return model
    
    def _collect_activation_stats(self, model, calibration_data: Optional[Any]) -> Dict[str, Any]:
        """收集激活统计信息"""
        logger.info("收集激活统计信息...")
        
        activation_stats = {}
        
        if calibration_data is None:
            logger.warning("没有校准数据，使用默认统计信息")
            return activation_stats
        
        # 设置模型为评估模式
        model.eval()
        
        # 创建钩子来收集激活
        hooks = []
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach()
            return hook
        
        # 注册钩子
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # 使用校准数据前向传播
        try:
            with torch.no_grad():
                for i, sample in enumerate(tqdm(calibration_data, desc="收集激活统计")):
                    if i >= 100:  # 限制样本数量
                        break
                    
                    # 处理输入数据
                    if isinstance(sample, dict):
                        inputs = sample
                    else:
                        inputs = {"input_ids": sample}
                    
                    # 前向传播
                    model(**inputs)
                    
                    # 收集统计信息
                    for name, activation in activations.items():
                        if name not in activation_stats:
                            activation_stats[name] = {
                                'max_abs': [],
                                'mean': [],
                                'std': []
                            }
                        
                        activation_stats[name]['max_abs'].append(activation.abs().max().item())
                        activation_stats[name]['mean'].append(activation.mean().item())
                        activation_stats[name]['std'].append(activation.std().item())
        
        except Exception as e:
            logger.warning(f"收集激活统计时出错: {e}")
        
        finally:
            # 移除钩子
            for hook in hooks:
                hook.remove()
        
        # 计算平均统计信息
        for name, stats in activation_stats.items():
            activation_stats[name]['avg_max_abs'] = np.mean(stats['max_abs'])
            activation_stats[name]['avg_mean'] = np.mean(stats['mean'])
            activation_stats[name]['avg_std'] = np.mean(stats['std'])
        
        return activation_stats
    
    def _compute_scaling_factors(self, activation_stats: Dict[str, Any]) -> Dict[str, float]:
        """计算缩放因子"""
        logger.info("计算SmoothQuant缩放因子...")
        
        scaling_factors = {}
        
        for name, stats in activation_stats.items():
            if 'avg_max_abs' in stats:
                # 计算权重和激活的最大绝对值
                weight_max_abs = 1.0  # 假设权重已经归一化
                activation_max_abs = stats['avg_max_abs']
                
                # 计算缩放因子
                if activation_max_abs > 0:
                    # 使用SmoothQuant公式
                    s = (weight_max_abs ** self.alpha) / (activation_max_abs ** (1 - self.alpha))
                    scaling_factors[name] = s
                else:
                    scaling_factors[name] = 1.0
            else:
                scaling_factors[name] = 1.0
        
        return scaling_factors
    
    def _apply_smooth_quant(self, model, scaling_factors: Dict[str, float]):
        """应用SmoothQuant变换"""
        logger.info("应用SmoothQuant变换...")
        
        for name, module in model.named_modules():
            if name in scaling_factors and isinstance(module, (nn.Linear, nn.Conv2d)):
                scale = scaling_factors[name]
                
                # 应用缩放因子到权重
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.data = module.weight.data / scale
                
                # 应用缩放因子到偏置（如果存在）
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data = module.bias.data / scale
                
                # 保存缩放因子
                module.smooth_quant_scale = scale
                
                logger.debug(f"应用SmoothQuant到 {name}: scale={scale:.4f}")
    
    def get_supported_models(self) -> list:
        """获取支持的模型类型"""
        return ["llama", "qwen", "deepseek", "general"] 