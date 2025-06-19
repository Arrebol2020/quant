"""
MinMax量化器实现 - 基础的线性量化方法，vLLM兼容版本
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from ..core.quantizer_registry import BaseQuantizer

logger = logging.getLogger(__name__)

class MinMaxQuantizer(BaseQuantizer):
    """MinMax量化器 - 基础的线性量化方法，vLLM兼容"""
    
    def __init__(self):
        self.name = "minmax"
    
    def quantize(
        self,
        model,
        bits: int = 8,
        group_size: int = 128,
        layer_wise: bool = False,
        target_layers: Optional[List[int]] = None,
        calibration_data: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """执行MinMax量化"""
        logger.info(f"开始MinMax量化: {bits}bit, group_size={group_size}")
        
        if layer_wise:
            return self._quantize_layer_wise(
                model, bits, group_size, target_layers, calibration_data, config
            )
        else:
            return self._quantize_full_model(
                model, bits, group_size, calibration_data, config
            )
    
    def _quantize_full_model(
        self,
        model,
        bits: int,
        group_size: int,
        calibration_data: Optional[Any],
        config: Optional[Dict[str, Any]]
    ):
        """全模型量化"""
        logger.info("执行全模型MinMax量化...")
        
        # 获取配置参数
        config = config or {}
        minmax_config = config.get('minmax', {})
        symmetric = minmax_config.get('symmetric', False)
        per_channel = minmax_config.get('per_channel', False)
        
        # 遍历所有层进行量化
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                logger.debug(f"量化层: {name}")
                self._quantize_layer_minmax(
                    module, bits, group_size, symmetric, per_channel
                )
        
        # 为vLLM兼容性添加必要的属性
        self._add_vllm_compatibility(model, bits, group_size, symmetric, per_channel)
        
        logger.info("MinMax量化完成，已添加vLLM兼容性")
        return model
    
    def _quantize_layer_wise(
        self,
        model,
        bits: int,
        group_size: int,
        target_layers: Optional[List[int]],
        calibration_data: Optional[Any],
        config: Optional[Dict[str, Any]]
    ):
        """逐层量化"""
        logger.info("执行逐层MinMax量化...")
        
        # 获取配置参数
        config = config or {}
        minmax_config = config.get('minmax', {})
        symmetric = minmax_config.get('symmetric', False)
        per_channel = minmax_config.get('per_channel', False)
        
        # 获取所有层
        layers = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                layers.append((name, module))
        
        # 过滤目标层
        if target_layers is not None:
            layers = [layers[i] for i in target_layers if i < len(layers)]
        
        # 逐层量化
        for name, layer in tqdm(layers, desc="量化层"):
            logger.info(f"量化层: {name}")
            self._quantize_layer_minmax(
                layer, bits, group_size, symmetric, per_channel
            )
        
        # 为vLLM兼容性添加必要的属性
        self._add_vllm_compatibility(model, bits, group_size, symmetric, per_channel)
        
        logger.info("逐层MinMax量化完成")
        return model
    
    def _quantize_layer_minmax(
        self, 
        layer, 
        bits: int, 
        group_size: int, 
        symmetric: bool = False,
        per_channel: bool = False
    ):
        """使用MinMax方法量化单个层"""
        if not hasattr(layer, 'weight') or layer.weight is None:
            return
        
        weight = layer.weight.data
        
        # 计算量化参数
        if per_channel:
            scale, zero_point = self._compute_per_channel_params(weight, bits, symmetric)
        elif group_size > 0:
            scale, zero_point = self._compute_group_params(weight, bits, group_size, symmetric)
        else:
            scale, zero_point = self._compute_global_params(weight, bits, symmetric)
        
        # 量化权重
        quantized_weight = self._quantize_weight_minmax(weight, scale, zero_point, bits, group_size)
        
        # 更新层权重
        layer.weight.data = quantized_weight
        
        # 保存量化参数（vLLM兼容格式）
        layer.scale = scale
        layer.zero_point = zero_point
        layer.bits = bits
        layer.quantization_method = "minmax"
        layer.symmetric = symmetric
        layer.per_channel = per_channel
        
        # 添加vLLM需要的属性
        layer.qweight = quantized_weight
        layer.scales = scale
        layer.zeros = zero_point
    
    def _compute_global_params(self, weight: torch.Tensor, bits: int, symmetric: bool = False) -> tuple:
        """计算全局量化参数"""
        if symmetric:
            # 对称量化
            w_max = weight.abs().max()
            w_min = -w_max
            
            qmin = -(2 ** (bits - 1))
            qmax = 2 ** (bits - 1) - 1
        else:
            # 非对称量化
            w_min = weight.min()
            w_max = weight.max()
            
            qmin = 0
            qmax = 2 ** bits - 1
        
        # 计算缩放因子和零点
        scale = (w_max - w_min) / (qmax - qmin)
        zero_point = qmin - w_min / scale if scale > 0 else 0
        
        return scale, zero_point
    
    def _compute_per_channel_params(self, weight: torch.Tensor, bits: int, symmetric: bool = False) -> tuple:
        """计算逐通道量化参数"""
        if len(weight.shape) < 2:
            return self._compute_global_params(weight, bits, symmetric)
        
        # 获取输出通道数
        out_channels = weight.shape[0]
        scales = []
        zero_points = []
        
        for i in range(out_channels):
            channel_weight = weight[i]
            
            if symmetric:
                # 对称量化
                w_max = channel_weight.abs().max()
                w_min = -w_max
                
                qmin = -(2 ** (bits - 1))
                qmax = 2 ** (bits - 1) - 1
            else:
                # 非对称量化
                w_min = channel_weight.min()
                w_max = channel_weight.max()
                
                qmin = 0
                qmax = 2 ** bits - 1
            
            # 计算缩放因子和零点
            scale = (w_max - w_min) / (qmax - qmin)
            zero_point = qmin - w_min / scale if scale > 0 else 0
            
            scales.append(scale)
            zero_points.append(zero_point)
        
        # 转换为张量
        scale_tensor = torch.tensor(scales, device=weight.device, dtype=weight.dtype)
        zero_point_tensor = torch.tensor(zero_points, device=weight.device, dtype=weight.dtype)
        
        return scale_tensor, zero_point_tensor
    
    def _compute_group_params(self, weight: torch.Tensor, bits: int, group_size: int, symmetric: bool = False) -> tuple:
        """按组计算量化参数"""
        # 重塑权重为组
        original_shape = weight.shape
        weight_flat = weight.view(-1)
        
        # 计算组数
        num_groups = (weight_flat.numel() + group_size - 1) // group_size
        
        scales = []
        zero_points = []
        
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min(start_idx + group_size, weight_flat.numel())
            group_weights = weight_flat[start_idx:end_idx]
            
            if symmetric:
                # 对称量化
                w_max = group_weights.abs().max()
                w_min = -w_max
                
                qmin = -(2 ** (bits - 1))
                qmax = 2 ** (bits - 1) - 1
            else:
                # 非对称量化
                w_min = group_weights.min()
                w_max = group_weights.max()
                
                qmin = 0
                qmax = 2 ** bits - 1
            
            # 计算缩放因子和零点
            scale = (w_max - w_min) / (qmax - qmin)
            zero_point = qmin - w_min / scale if scale > 0 else 0
            
            scales.append(scale)
            zero_points.append(zero_point)
        
        # 转换为张量
        scale_tensor = torch.tensor(scales, device=weight.device, dtype=weight.dtype)
        zero_point_tensor = torch.tensor(zero_points, device=weight.device, dtype=weight.dtype)
        
        return scale_tensor, zero_point_tensor
    
    def _quantize_weight_minmax(
        self, 
        weight: torch.Tensor, 
        scale, 
        zero_point, 
        bits: int, 
        group_size: int
    ) -> torch.Tensor:
        """使用MinMax方法量化权重"""
        if isinstance(scale, torch.Tensor) and len(scale) > 1:
            if group_size > 0:
                return self._quantize_weight_group_minmax(weight, scale, zero_point, bits, group_size)
            else:
                return self._quantize_weight_per_channel_minmax(weight, scale, zero_point, bits)
        else:
            return self._quantize_weight_global_minmax(weight, scale, zero_point, bits)
    
    def _quantize_weight_global_minmax(
        self, 
        weight: torch.Tensor, 
        scale: float, 
        zero_point: float, 
        bits: int
    ) -> torch.Tensor:
        """全局量化权重"""
        # 应用量化
        quantized = torch.round(weight / scale + zero_point)
        
        # 限制范围
        if bits == 8:
            quantized = torch.clamp(quantized, 0, 255)
        elif bits == 4:
            quantized = torch.clamp(quantized, 0, 15)
        elif bits == 2:
            quantized = torch.clamp(quantized, 0, 3)
        else:
            qmax = 2 ** bits - 1
            quantized = torch.clamp(quantized, 0, qmax)
        
        # 反量化
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def _quantize_weight_per_channel_minmax(
        self, 
        weight: torch.Tensor, 
        scale: torch.Tensor, 
        zero_point: torch.Tensor, 
        bits: int
    ) -> torch.Tensor:
        """逐通道量化权重"""
        if len(weight.shape) < 2:
            return self._quantize_weight_global_minmax(weight, scale[0], zero_point[0], bits)
        
        # 获取输出通道数
        out_channels = weight.shape[0]
        quantized_weight = torch.zeros_like(weight)
        
        for i in range(out_channels):
            channel_weight = weight[i]
            channel_scale = scale[i]
            channel_zero_point = zero_point[i]
            
            # 量化
            quantized = torch.round(channel_weight / channel_scale + channel_zero_point)
            
            # 限制范围
            if bits == 8:
                quantized = torch.clamp(quantized, 0, 255)
            elif bits == 4:
                quantized = torch.clamp(quantized, 0, 15)
            elif bits == 2:
                quantized = torch.clamp(quantized, 0, 3)
            else:
                qmax = 2 ** bits - 1
                quantized = torch.clamp(quantized, 0, qmax)
            
            # 反量化
            dequantized = (quantized - channel_zero_point) * channel_scale
            quantized_weight[i] = dequantized
        
        return quantized_weight
    
    def _quantize_weight_group_minmax(
        self, 
        weight: torch.Tensor, 
        scale: torch.Tensor, 
        zero_point: torch.Tensor, 
        bits: int, 
        group_size: int
    ) -> torch.Tensor:
        """按组量化权重"""
        weight_flat = weight.view(-1)
        quantized_flat = torch.zeros_like(weight_flat)
        
        num_groups = len(scale)
        
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min(start_idx + group_size, weight_flat.numel())
            
            group_weights = weight_flat[start_idx:end_idx]
            group_scale = scale[i]
            group_zero_point = zero_point[i]
            
            # 量化
            quantized = torch.round(group_weights / group_scale + group_zero_point)
            
            # 限制范围
            if bits == 8:
                quantized = torch.clamp(quantized, 0, 255)
            elif bits == 4:
                quantized = torch.clamp(quantized, 0, 15)
            elif bits == 2:
                quantized = torch.clamp(quantized, 0, 3)
            else:
                qmax = 2 ** bits - 1
                quantized = torch.clamp(quantized, 0, qmax)
            
            # 反量化
            dequantized = (quantized - group_zero_point) * group_scale
            quantized_flat[start_idx:end_idx] = dequantized
        
        return quantized_flat.view(weight.shape)
    
    def _add_vllm_compatibility(self, model, bits: int, group_size: int, symmetric: bool, per_channel: bool):
        """为模型添加vLLM兼容性"""
        logger.info("添加vLLM兼容性...")
        
        # 添加vLLM需要的配置属性
        if hasattr(model, 'config'):
            model.config.quantization_config = {
                "quantization_method": "minmax",
                "bits": bits,
                "group_size": group_size,
                "symmetric": symmetric,
                "per_channel": per_channel
            }
        
        # 为所有量化层添加vLLM兼容属性
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                # 确保量化参数存在
                if not hasattr(module, 'qweight'):
                    module.qweight = module.weight.data
                if not hasattr(module, 'scales'):
                    module.scales = getattr(module, 'scale', torch.tensor(1.0))
                if not hasattr(module, 'zeros'):
                    module.zeros = getattr(module, 'zero_point', torch.tensor(0.0))
                
                # 添加vLLM需要的其他属性
                module.bits = getattr(module, 'bits', bits)
                module.group_size = group_size
                module.quantization_method = "minmax"
                module.symmetric = symmetric
                module.per_channel = per_channel
    
    def get_supported_bits(self) -> List[int]:
        """获取支持的量化位数"""
        return [2, 3, 4, 8, 16]
    
    def get_supported_group_sizes(self) -> List[int]:
        """获取支持的组大小"""
        return [0, 32, 64, 128, 256]  # 0表示不使用分组 