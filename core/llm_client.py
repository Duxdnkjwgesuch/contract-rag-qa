"""
大模型客户端 —— 统一封装 ollama OpenAI SiliconFlow 等大模型客户端
"""
import logging
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from config.settings import settings

logger = logging.getLogger(__name__)#记录日志
#这个类是调用LLM客户端的
class LLMClient:
    def __init__(self):
        self._llm = None#_llm 默认是None
        self._initialize_client() #初始化大模型客户端的方法

    def _initialize_client(self):
        """初始化大模型客户端"""
        try:
            if settings.LLM_PROVIDER == "ollama":#如果设置的模型是ollama 就初始化ollama
                self._llm = ChatOllama(
                    model=settings.OLLAMA_MODEL_NAME,
                    base_url=settings.OLLAMA_BASE_URL,
                    temperature=0.1,
                    verbose=True
                )
                logger.info(f"Ollama 客户端已初始化: {settings.OLLAMA_MODEL_NAME}")
            elif settings.LLM_PROVIDER == "openai":#如果是openai 就初始化openai
                self._llm = ChatOpenAI(
                    model=settings.OPENAI_MODEL_NAME,
                    api_key=settings.OPENAI_API_KEY,
                    base_url=settings.OPENAI_BASE_URL,
                    temperature=0.1,
                    verbose=True
                )
                logger.info(f"OpenAI 客户端已初始化: {settings.OPENAI_MODEL_NAME}")
            else:
                self._llm = ChatOpenAI(#如果都不是就初始化硅基流动的
                    model=settings.SILICONFLOW_MODEL_NAME,
                    api_key=settings.SILICONFLOW_API_KEY,
                    base_url=settings.SILICONFLOW_BASE_URL,
                    temperature=0.1,
                    verbose=True
                )
                logger.info(f"SiliconFlow 客户端已初始化: {settings.SILICONFLOW_MODEL_NAME}")
        except Exception as e:
            logger.error(f"初始化大模型客户端失败: {e}")#失败了就提醒初始化失败 返回异常
            raise

    def get_llm(self):#获取当前大模型实例
        """返回大模型实例"""
        return self._llm

    def change_provider(self, provider: str, **kwargs):#切换大模型方法
        """
        切换大模型提供商
        """
        settings.LLM_PROVIDER = provider
        for key, value in kwargs.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        self._initialize_client()
        logger.info(f"大模型提供商已切换为: {provider}")

"""实例化大模型"""
_llm_client = None#默认_llm_client为None

def get_llm():#返回当前大模型实例
    """获取大模型实例"""
    global _llm_client
    if _llm_client is None:#如果是None就创建LLMClient类的对象 此时_llm_client就不为None了 返回当前的大模型
        _llm_client = LLMClient()
    return _llm_client.get_llm()
