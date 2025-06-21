#!/usr/bin/env python3
"""
大模型量化工具主入口
支持多种量化算法和模型架构
"""

import click
import yaml
import logging
from pathlib import Path
from typing import Optional, List

from src.core.quantization_manager import QuantizationManager
from src.utils.config import load_config
from src.app import init_app

@click.command()
@click.option('--model-path', required=True, help='原始模型路径')
@click.option('--output-path', required=True, help='量化后模型输出路径')
@click.option('--method', default='gptq', help='量化方法 (gptq/awq/minmax/hqq)')
@click.option('--bits', default=4, help='量化位数 (2/3/4/8/16)')
@click.option('--group-size', default=128, help='量化组大小')
@click.option('--layer-wise', is_flag=True, help='是否启用逐层量化')
@click.option('--outlier-suppression', default=None, help='离群值抑制算法 (smooth_quant等)')
@click.option('--config', default=None, help='配置文件路径')
@click.option('--layers', default=None, help='指定要量化的层 (逗号分隔)')
@click.option('--calibration-dataset', default=None, help='校准数据集路径')
@click.option('--verbose', is_flag=True, help='详细输出')
def main(
    model_path: str,
    output_path: str,
    method: str,
    bits: int,
    group_size: int,
    layer_wise: bool,
    outlier_suppression: Optional[str],
    config: Optional[str],
    layers: Optional[str],
    calibration_dataset: Optional[str],
    verbose: bool
):
    """大模型量化工具"""
    
    # 初始化应用
    log_level = logging.DEBUG if verbose else logging.INFO
    init_app(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("开始大模型量化...")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"输出路径: {output_path}")
    logger.info(f"量化方法: {method}")
    logger.info(f"量化位数: {bits}")
    
    # 加载配置
    if config:
        config_dict = load_config(config)
    else:
        config_dict = {}
    
    # 解析层列表
    layer_list = None
    if layers:
        layer_list = [int(x.strip()) for x in layers.split(',')]
    
    # 创建量化管理器
    manager = QuantizationManager(
        model_path=model_path,
        output_path=output_path,
        method=method,
        bits=bits,
        group_size=group_size,
        layer_wise=layer_wise,
        outlier_suppression=outlier_suppression,
        config=config_dict,
        target_layers=layer_list,
        calibration_dataset=calibration_dataset
    )
    
    # 执行量化
    try:
        manager.quantize()
        logger.info("量化完成!")
    except Exception as e:
        logger.error(f"量化失败: {e}")
        raise

if __name__ == '__main__':
    main() 