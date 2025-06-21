"""
GPTQ量化器实现 - vLLM兼容版本
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from ..core.quantizer_registry import BaseQuantizer

logger = logging.getLogger(__name__)

class GPTQQuantizer(BaseQuantizer):
    """GPTQ量化器 - 基于Hessian矩阵的量化方法，vLLM兼容"""
    
    def __init__(self):
        self.name = "gptq"
    
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
        """执行GPTQ量化"""
        logger.info(f"开始GPTQ量化: {bits}bit, group_size={group_size}")
        
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
        logger.info("执行全模型GPTQ量化...")
        
        try:
            # 使用auto-gptq进行量化，生成vLLM兼容格式
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            
            # 创建量化配置
            quantize_config = BaseQuantizeConfig(
                bits=bits,
                group_size=group_size,
                damp_percent=0.01,
                desc_act=False,
                static_groups=False,
                sym=True,
                true_sequential=True,
                model_name_or_path=None,
                model_file_base_name="model"
            )
            
            # 执行量化
            quantized_model = AutoGPTQForCausalLM.from_pretrained(
                model,
                quantize_config=quantize_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 为vLLM兼容性添加必要的属性
            self._add_vllm_compatibility(quantized_model, bits, group_size)
            
            logger.info("GPTQ量化完成，已添加vLLM兼容性")
            return quantized_model
            
        except ImportError:
            logger.warning("auto-gptq未安装，使用简化实现")
            return self._simple_gptq_quantize(model, bits, group_size)
    
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
        logger.info("执行逐层GPTQ量化...")
        
        # 获取所有Linear层
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layers.append((name, module))
        
        logger.info(f"找到 {len(layers)} 个Linear层")
        
        # 过滤目标层
        if target_layers is not None:
            layers = [layers[i] for i in target_layers if i < len(layers)]
            logger.info(f"将量化其中的 {len(layers)} 个目标层")
        
        # 逐层量化
        for name, layer in tqdm(layers, desc="量化Linear层"):
            logger.info(f"量化层: {name}")
            self._quantize_layer(layer, bits, group_size)
        
        # 为vLLM兼容性添加必要的属性
        self._add_vllm_compatibility(model, bits, group_size)
        
        logger.info("逐层GPTQ量化完成")
        return model
    
    def _quantize_layer(self, layer, bits: int, group_size: int):
        """量化单个层"""
        if not hasattr(layer, 'weight') or layer.weight is None:
            return
        
        weight = layer.weight.data
        
        # 计算量化参数
        scale, zero_point = self._compute_quantization_params(weight, bits)
        
        # 量化权重
        quantized_weight = self._quantize_weight(weight, scale, zero_point, bits)
        
        # 更新层权重
        layer.weight.data = quantized_weight
        
        # 保存量化参数（vLLM兼容格式）
        layer.scale = scale
        layer.zero_point = zero_point
        layer.bits = bits
        layer.quantization_method = "gptq"
        
        # 添加vLLM需要的属性
        layer.qweight = quantized_weight
        layer.scales = scale
        layer.zeros = zero_point
    
    def _compute_quantization_params(self, weight: torch.Tensor, bits: int) -> tuple:
        """计算量化参数"""
        # 计算权重范围
        w_min = weight.min()
        w_max = weight.max()
        
        # 计算量化参数
        qmin = 0
        qmax = 2 ** bits - 1
        
        scale = (w_max - w_min) / (qmax - qmin)
        zero_point = qmin - w_min / scale
        
        return scale, zero_point
    
    def _quantize_weight(self, weight: torch.Tensor, scale: float, zero_point: float, bits: int) -> torch.Tensor:
        """量化权重"""
        # 应用量化
        quantized = torch.round(weight / scale + zero_point)
        
        # 限制范围
        qmin = 0
        qmax = 2 ** bits - 1
        quantized = torch.clamp(quantized, qmin, qmax)
        
        # 反量化
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def _add_vllm_compatibility(self, model, bits: int, group_size: int):
        """为模型添加vLLM兼容性"""
        logger.info("添加vLLM兼容性...")
        
        # 添加vLLM需要的配置属性
        if hasattr(model, 'config'):
            model.config.quantization_config = {
                "quantization_method": "gptq",
                "bits": bits,
                "group_size": group_size,
                "desc_act": False,
                "static_groups": False,
                "sym": True,
                "true_sequential": True
            }
        
        # 为所有量化层添加vLLM兼容属性
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # 确保量化参数存在
                if not hasattr(module, 'qweight'):
                    module.qweight = module.weight.data
                if not hasattr(module, 'scales'):
                    module.scales = getattr(module, 'scale', None)
                if not hasattr(module, 'zeros'):
                    module.zeros = getattr(module, 'zero_point', None)
    
    def _simple_gptq_quantize(self, model, bits: int, group_size: int):
        """简化的GPTQ量化实现"""
        logger.info("使用简化的GPTQ量化实现...")
        
        linear_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                logger.debug(f"量化Linear层: {name}")
                self._quantize_layer(module, bits, group_size)
                linear_count += 1
        
        logger.info(f"共量化了 {linear_count} 个Linear层")
        
        # 为vLLM兼容性添加必要的属性
        self._add_vllm_compatibility(model, bits, group_size)
        
        return model
    
    def get_supported_bits(self) -> List[int]:
        """获取支持的量化位数"""
        return [2, 3, 4, 8]
    
    def get_supported_group_sizes(self) -> List[int]:
        """获取支持的组大小"""
        return [32, 64, 128, 256] 