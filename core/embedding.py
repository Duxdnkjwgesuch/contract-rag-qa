"""
embedding模型加载模块——使用单例模式，支持本地缓存 与 HuggingFace自动下载

单例模式：是一种设计模式，它保证一个类只有一个实例，并提供一个全局访问点。
在Python中，我们可以使用类方法来实现单例模式。 cls.__instance 是一个类变量，用于存储类的唯一实例。
目的：避免重复加载与初始化，提高性能 和 资源利用率。
"""

import logging
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)#日志开启

class EmbeddingModel:
    """嵌入模型 单例模式"""
    _instance = None#规定_instance默认是None
    # __new__是一个静态方法，用于创建类的实例
    def __new__(cls):
        if cls._instance is None:#如果cls._instance是None 就创建 如果不是None就复用 （单例模式）
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化嵌入模型"""
        # 尝试从本地先加载，本地没有再从HuggingFace上面下载
        model_name = settings.EMBEDDING_MODEL_NAME
        # .replace("/", "_") 是因为 Qwen/Qwen2.5-7B-Instruct 这个模型名中包含了 / ，会导致路径错误
        local_model_path = settings.MODELS_DIR / model_name.replace("/", "_")
        if local_model_path.exists():#如果本地模型存在 从本地下载
            logger.info(f"从本地加载嵌入模型: {local_model_path}")
            model_path = str(local_model_path)
        else:
            logger.info(f"从HuggingFace下载嵌入模型: {model_name}")#从HF上下载
            model_path = model_name


        try:
            self.model = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={"device": settings.EMBEDDING_DEVICE},
                encode_kwargs={
                        "normalize_embeddings": True,
                        "batch_size": 32,
                    }
            )#设置HF模型的参数
            logger.info(f"嵌入模型已初始化: {model_name}")#提示加载成功
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {e}")#提示加载失败
            raise

    def get_model(self):#得到当前模型的实例
        """ 返回嵌入模型实例 """
        return self.model


def get_embedding_model():#得到当前向量模型的实例
    """ 获取嵌入模型实例 """
    return EmbeddingModel().get_model()
