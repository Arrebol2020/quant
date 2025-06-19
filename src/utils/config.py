"""
配置管理工具
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")

def save_config(config: Dict[str, Any], config_path: str):
    """保存配置文件"""
    config_path = Path(config_path)
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """合并配置"""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged

def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        "quantization": {
            "method": "gptq",
            "bits": 4,
            "group_size": 128,
            "layer_wise": False,
            "outlier_suppression": None
        },
        "model": {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": "float16"
        },
        "calibration": {
            "dataset": None,
            "num_samples": 100,
            "batch_size": 1
        },
        "output": {
            "save_format": "safetensors",
            "save_quantization_config": True
        }
    } 