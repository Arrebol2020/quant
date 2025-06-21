"""
应用程序初始化模块
负责集中管理所有初始化操作
"""

import logging
from typing import Optional

from .utils.logging import setup_logging
from .registry_init import register_all_components

logger = logging.getLogger(__name__)

class App:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._setup_completed = False

    def setup(self, log_level: Optional[int] = None):
        """
        初始化应用程序
        
        Args:
            log_level: 日志级别，默认为 None，表示使用 INFO 级别
        """
        if self._setup_completed:
            logger.warning("App.setup() 被多次调用")
            return

        # 设置日志
        if log_level is None:
            log_level = logging.INFO
        setup_logging(log_level)
        
        # 注册所有组件
        logger.info("正在初始化应用组件...")
        register_all_components()
        
        self._setup_completed = True
        logger.info("应用程序初始化完成")

    @property
    def is_initialized(self) -> bool:
        """检查应用是否已经完成初始化"""
        return self._setup_completed

def init_app(log_level: Optional[int] = None) -> App:
    """
    初始化应用程序的便捷函数
    
    Args:
        log_level: 日志级别，默认为 None，表示使用 INFO 级别
        
    Returns:
        App: 应用程序实例
    """
    app = App()
    app.setup(log_level)
    return app 