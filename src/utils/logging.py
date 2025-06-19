"""
日志管理工具
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
):
    """设置日志配置"""
    
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 创建格式化器
    formatter = logging.Formatter(format_string)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 添加文件处理器（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('datasets').setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志记录器"""
    return logging.getLogger(name)

def log_quantization_info(logger: logging.Logger, info: dict):
    """记录量化信息"""
    logger.info("=" * 50)
    logger.info("量化配置信息:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 50)

def log_model_info(logger: logging.Logger, model_info: dict):
    """记录模型信息"""
    logger.info("=" * 50)
    logger.info("模型信息:")
    for key, value in model_info.items():
        if isinstance(value, (int, float)) and value > 1000:
            # 格式化大数字
            if value > 1e9:
                formatted_value = f"{value/1e9:.2f}B"
            elif value > 1e6:
                formatted_value = f"{value/1e6:.2f}M"
            elif value > 1e3:
                formatted_value = f"{value/1e3:.2f}K"
            else:
                formatted_value = str(value)
            logger.info(f"  {key}: {formatted_value}")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("=" * 50) 