"""
量化管理器 - 协调整个量化过程
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .model_registry import ModelRegistry
from .quantizer_registry import QuantizerRegistry
from .outlier_suppression_registry import OutlierSuppressionRegistry
from ..utils.config import merge_configs

logger = logging.getLogger(__name__)

class QuantizationManager:
    """量化管理器 - 协调模型加载、量化和保存"""
    
    def __init__(
        self,
        model_path: str,
        output_path: str,
        method: str = "gptq",
        bits: int = 4,
        group_size: int = 128,
        layer_wise: bool = False,
        outlier_suppression: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        target_layers: Optional[List[int]] = None,
        calibration_dataset: Optional[str] = None
    ):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.method = method
        self.bits = bits
        self.group_size = group_size
        self.layer_wise = layer_wise
        self.outlier_suppression = outlier_suppression
        self.config = config or {}
        self.target_layers = target_layers
        self.calibration_dataset = calibration_dataset
        
        # 创建输出目录
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self):
        """初始化各个组件"""
        logger.info("初始化量化组件...")
        
        # 加载模型适配器
        self.model_adapter = ModelRegistry.get_adapter(self.model_path)
        logger.info(f"使用模型适配器: {self.model_adapter.__class__.__name__}")
        
        # 加载量化器
        self.quantizer = QuantizerRegistry.get_quantizer(self.method)
        logger.info(f"使用量化器: {self.quantizer.__class__.__name__}")
        
        # 加载离群值抑制器（如果指定）
        self.outlier_suppressor = None
        if self.outlier_suppression:
            self.outlier_suppressor = OutlierSuppressionRegistry.get_suppressor(
                self.outlier_suppression
            )
            logger.info(f"使用离群值抑制器: {self.outlier_suppressor.__class__.__name__}")
    
    def quantize(self):
        """执行量化过程"""
        logger.info("开始量化过程...")
        
        # 1. 加载模型
        logger.info("加载原始模型...")
        model = self.model_adapter.load_model(self.model_path)
        
        # 2. 准备校准数据
        calibration_data = None
        if self.calibration_dataset:
            logger.info("加载校准数据...")
            calibration_data = self.model_adapter.load_calibration_data(
                self.calibration_dataset
            )
        
        # 3. 应用离群值抑制（如果启用）
        if self.outlier_suppressor:
            logger.info("应用离群值抑制...")
            model = self.outlier_suppressor.apply(model, calibration_data)
        
        # 4. 执行量化
        logger.info(f"开始{self.method}量化...")
        quantized_model = self.quantizer.quantize(
            model=model,
            bits=self.bits,
            group_size=self.group_size,
            layer_wise=self.layer_wise,
            target_layers=self.target_layers,
            calibration_data=calibration_data,
            config=self.config
        )
        
        # 5. 保存量化后的模型
        logger.info("保存量化后的模型...")
        self.model_adapter.save_model(quantized_model, self.output_path)
        
        # 6. 生成量化报告
        self._generate_report()
        
        logger.info("量化过程完成!")
    
    def _generate_report(self):
        """生成量化报告"""
        logger.info("生成量化报告...")
        
        report = {
            "model_path": str(self.model_path),
            "output_path": str(self.output_path),
            "method": self.method,
            "bits": self.bits,
            "group_size": self.group_size,
            "layer_wise": self.layer_wise,
            "outlier_suppression": self.outlier_suppression,
            "target_layers": self.target_layers,
            "model_adapter": self.model_adapter.__class__.__name__,
            "quantizer": self.quantizer.__class__.__name__,
            "outlier_suppressor": self.outlier_suppressor.__class__.__name__ if self.outlier_suppressor else None
        }
        
        # 保存报告
        report_path = self.output_path / "quantization_report.yaml"
        import yaml
        with open(report_path, 'w', encoding='utf-8') as f:
            yaml.dump(report, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"量化报告已保存到: {report_path}") 