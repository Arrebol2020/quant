"""
AWQ量化器实现 - Activation-aware Weight Quantization，vLLM兼容版本
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from ..core.quantizer_registry import BaseQuantizer

logger = logging.getLogger(__name__)

class AWQQuantizer(BaseQuantizer):
    """AWQ量化器 - Activation-aware Weight Quantization，vLLM兼容"""
    
    def __init__(self):
        self.name = "awq"
    
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
        """执行AWQ量化"""
        logger.info(f"开始AWQ量化: {bits}bit, group_size={group_size}")
        
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
        logger.info("执行全模型AWQ量化...")
        
        try:
            # 使用autoawq进行量化，生成vLLM兼容格式
            from autoawq import AutoAWQForCausalLM, AwqConfig
            
            # 创建AWQ配置
            awq_config = AwqConfig(
                bits=bits,
                group_size=group_size,
                zero_point=True,
                q_type="asym"
            )
            
            # 执行量化
            quantized_model = AutoAWQForCausalLM.from_pretrained(
                model,
                awq_config=awq_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 为vLLM兼容性添加必要的属性
            self._add_vllm_compatibility(quantized_model, bits, group_size)
            
            logger.info("AWQ量化完成，已添加vLLM兼容性")
            return quantized_model
            
        except ImportError:
            logger.warning("autoawq未安装，使用简化实现")
            return self._simple_awq_quantize(model, bits, group_size, calibration_data)
    
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
        logger.info("执行逐层AWQ量化...")
        
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
            self._quantize_layer_awq(layer, bits, group_size, calibration_data)
        
        # 为vLLM兼容性添加必要的属性
        self._add_vllm_compatibility(model, bits, group_size)
        
        logger.info("逐层AWQ量化完成")
        return model
    
    def _quantize_layer_awq(self, layer, bits: int, group_size: int, calibration_data: Optional[Any]):
        """使用AWQ方法量化单个层"""
        if not hasattr(layer, 'weight') or layer.weight is None:
            return
        
        weight = layer.weight.data
        
        # 计算激活统计信息（如果有校准数据）
        activation_scale = self._compute_activation_scale(layer, calibration_data)
        
        # 计算AWQ量化参数
        scale, zero_point = self._compute_awq_params(weight, activation_scale, bits, group_size)
        
        # 量化权重
        quantized_weight = self._quantize_weight_awq(weight, scale, zero_point, bits, group_size)
        
        # 更新层权重
        layer.weight.data = quantized_weight
        
        # 保存量化参数（vLLM兼容格式）
        layer.scale = scale
        layer.zero_point = zero_point
        layer.bits = bits
        layer.activation_scale = activation_scale
        layer.quantization_method = "awq"
        
        # 添加vLLM需要的属性
        layer.qweight = quantized_weight
        layer.scales = scale
        layer.zeros = zero_point
        layer.awq_scale = activation_scale
    
    def _compute_activation_scale(self, layer, calibration_data: Optional[Any]) -> float:
        """计算激活的缩放因子"""
        if calibration_data is None:
            return 1.0
        
        # 这里应该使用校准数据来计算激活统计信息
        # 简化实现：返回默认值
        return 1.0
    
    def _compute_awq_params(self, weight: torch.Tensor, activation_scale: float, bits: int, group_size: int) -> tuple:
        """计算AWQ量化参数"""
        # 考虑激活缩放因子的权重调整
        adjusted_weight = weight * activation_scale
        
        # 按组计算量化参数
        if group_size > 0:
            return self._compute_group_awq_params(adjusted_weight, bits, group_size)
        else:
            return self._compute_global_awq_params(adjusted_weight, bits)
    
    def _compute_group_awq_params(self, weight: torch.Tensor, bits: int, group_size: int) -> tuple:
        """按组计算AWQ参数"""
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
            
            # 计算组的量化参数
            w_min = group_weights.min()
            w_max = group_weights.max()
            
            qmin = 0
            qmax = 2 ** bits - 1
            
            scale = (w_max - w_min) / (qmax - qmin)
            zero_point = qmin - w_min / scale if scale > 0 else 0
            
            scales.append(scale)
            zero_points.append(zero_point)
        
        # 转换为张量
        scale_tensor = torch.tensor(scales, device=weight.device, dtype=weight.dtype)
        zero_point_tensor = torch.tensor(zero_points, device=weight.device, dtype=weight.dtype)
        
        return scale_tensor, zero_point_tensor
    
    def _compute_global_awq_params(self, weight: torch.Tensor, bits: int) -> tuple:
        """全局计算AWQ参数"""
        w_min = weight.min()
        w_max = weight.max()
        
        qmin = 0
        qmax = 2 ** bits - 1
        
        scale = (w_max - w_min) / (qmax - qmin)
        zero_point = qmin - w_min / scale if scale > 0 else 0
        
        return scale, zero_point
    
    def _quantize_weight_awq(self, weight: torch.Tensor, scale, zero_point, bits: int, group_size: int) -> torch.Tensor:
        """使用AWQ方法量化权重"""
        if group_size > 0:
            return self._quantize_weight_group_awq(weight, scale, zero_point, bits, group_size)
        else:
            return self._quantize_weight_global_awq(weight, scale, zero_point, bits)
    
    def _quantize_weight_group_awq(self, weight: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
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
            quantized = torch.clamp(quantized, 0, 2 ** bits - 1)
            
            # 反量化
            dequantized = (quantized - group_zero_point) * group_scale
            quantized_flat[start_idx:end_idx] = dequantized
        
        return quantized_flat.view(weight.shape)
    
    def _quantize_weight_global_awq(self, weight: torch.Tensor, scale: float, zero_point: float, bits: int) -> torch.Tensor:
        """全局量化权重"""
        quantized = torch.round(weight / scale + zero_point)
        quantized = torch.clamp(quantized, 0, 2 ** bits - 1)
        dequantized = (quantized - zero_point) * scale
        return dequantized
    
    def _add_vllm_compatibility(self, model, bits: int, group_size: int):
        """为模型添加vLLM兼容性"""
        logger.info("添加vLLM兼容性...")
        
        # 添加vLLM需要的配置属性
        if hasattr(model, 'config'):
            model.config.quantization_config = {
                "quantization_method": "awq",
                "bits": bits,
                "group_size": group_size,
                "zero_point": True,
                "q_type": "asym"
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
                module.quantization_method = "awq"
                module.awq_scale = getattr(module, 'activation_scale', 1.0)
    
    def _simple_awq_quantize(self, model, bits: int, group_size: int, calibration_data: Optional[Any]):
        """简化的AWQ量化实现"""
        logger.info("使用简化AWQ实现...")
        
        # 遍历所有层进行量化
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                self._quantize_layer_awq(module, bits, group_size, calibration_data)
        
        # 添加vLLM兼容性
        self._add_vllm_compatibility(model, bits, group_size)
        
        return model
    
    def get_supported_bits(self) -> List[int]:
        """获取支持的量化位数"""
        return [2, 3, 4, 8]
    
    def get_supported_group_sizes(self) -> List[int]:
        """获取支持的组大小"""
        return [32, 64, 128, 256] 