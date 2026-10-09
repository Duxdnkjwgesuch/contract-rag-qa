"""
检索模块 —— 支持相似度检索 + 可选的重排序
主要功能：
    1. 基于向量相似度进行文档检索
    2. 可选的文档重排序（使用Reranker）
        使用重排序的原因是为了提高检索结果的相关性和准确性。
        这属于对RAG技术的优化，主要就是为了提高检索结果的质量。
        但是会增加一些额外的开销。
"""
import logging
from config.settings import settings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents.compressor import BaseDocumentCompressor
from core.vector_store import get_vector_store_manager

logger = logging.getLogger(__name__)#记录日志

class CrossEncoderReranker(BaseDocumentCompressor):#重排序的交叉编码器
    """基于交叉编码器的重排序器"""
    model_name: str#模型名称
    top_k: int#筛选数
    model: object = None  # 延迟加载模型
    def __init__(self, model_name: str, top_k: int = 5):
        """初始化重排序器"""
        super().__init__()
        self.model_name = model_name or settings.RERANKER_MODEL_NAME#加载重排序模型 和 top——k的值
        self.top_k = top_k
        # 延迟初始化模型，避免启动时加载耗时！！！！！！

    def _initialize_model(self):#这个方法是在给 CrossEncoderReranker 做延迟初始化 重排序模型不在对象创建时加载，而是在第一次真正需要重排序时才加载。
        """初始化交叉编码器模型，使用时才第一次加载"""
        if self.model is None:#如果模型是空的 才加载这个方法
            try:
                from sentence_transformers import CrossEncoder#第一次调用时导入并创建模型
                self.model = CrossEncoder(self.model_name)#将CrossEncoder赋值给model
                logger.info(f"交叉编码器模型已初始化: {self.model_name}")#记录日志 交叉编码器模型初始化成功
            except Exception as e:
                logger.error(f"初始化交叉编码器模型失败: {str(e)}")#否则抛出异常 模型赋值为None
                self.model = None

    def compress_documents(self, documents, query, **kwargs):#交叉编码器的核心方法 相关性排序 接收用户问题和一批候选文档，使用 CrossEncoder 判断每个“问题 + 文档”组合的相关性，然后按照相关性从高到低排序，返回最相关的 top_k 个文档。
        """重排序核心方法：对文档列表按与查询query的相关性排序"""
        if not documents:
            return []#如果没有相关的文档列表就返回一个空列表
        self._initialize_model()#初始化重排序模型
        if self.model is None:
            # 模型加载失败，直接返回原文档，不做排序
            return documents[:self.top_k]
        try:
            # 构建 查询-文档对 理解为qa对
            pairs = [[query, doc.page_content] for doc in documents]
            # 使用交叉编码器模型 进行打分
            scores = self.model.predict(pairs)
            # 按分数降序排序
            sorted_docs = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True
            )
            # 返回前top_k个文档
            return [doc for doc, _ in sorted_docs[:self.top_k]]
        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")#出错抛出异常 并记录日志
            return documents[:self.top_k]#返回原文档



class RAGRetriever:#检索器类 作用是：它负责 RAG 流程中的“找资料”环节。先从向量数据库中召回候选文档，再可选地使用 CrossEncoder 重排序，最后把质量更高、与问题更相关的文档交给大模型作为上下文。
    """RAG检索器，封装向量检索和（可选的）重排序功能"""
    def __init__(self):
        self.vector_store = get_vector_store_manager()#初始化加载 向量数据库
        self.reranker = self._initialize_reranker()#初始化加载 重排序功能（交叉编码器）

    def _initialize_reranker(self):#初始化Reranker
        """初始化Reranker"""
        if not settings.USE_RERANKER:#如果没有Reranker 就返回None
            logging.info("未启用Reranker")
            return None
        try:
            # 尝试本地加载，本地没有的话再huggingface下载
            model_name = settings.RERANKER_MODEL_NAME
            local_model_path = settings.MODELS_DIR / model_name.replace("/", "_")
            if local_model_path.exists():
                model_path = str(local_model_path)
                logger.info(f"从本地加载Reranker模型: {local_model_path}")
            else:
                model_path = model_name
                logger.info(f"从HuggingFace下载Reranker模型: {model_name}")
            # 创建重排序实例
            reranker = CrossEncoderReranker(
                model_name=model_path,
                top_k=settings.SEARCH_TOP_K
            )
            logger.info(f"Reranker已初始化: {settings.RERANKER_MODEL_NAME}")#Reranker初始化成功
            return reranker
        except Exception as e:
            logging.error(f"初始化Reranker失败: {str(e)}")#失败返回None
            return None

    def retrieve(self, query: str, top_k: int = None, filter_dict: dict = None, return_score: bool = False) -> list:
        """
        检索文档
        :param query: 查询文本
        :param top_k: 返回的文档数量
        :param filter_dict: 过滤条件
        :param return_score: 是否返回相关性分数
        :return: 检索到的文档列表
        """
        top_k = top_k or settings.SEARCH_TOP_K #默认是5
        # 1. 执行向量检索 向量召回 ：包含对应的文档片段名 和 对应的 分数 通常分数越小表示越相似
        docs_with_score = self.vector_store.similarity_search_with_score(
            query,
            k=top_k,
            filter_dict=filter_dict
        )
        # 2. 如果启用了Reranker，则进行重排序
        if self.reranker:
            raw_docs = [doc for doc, _ in docs_with_score]#先从元组中取出文档（所有文档名）
            ranked_docs = self.reranker.compress_documents(raw_docs, query)#然后调用 将用户的输入与这些文档片段进行相关性打分，再按分数排序。
            # 重新匹配原分数 这段代码的意思是，对每个重排序后的文档，回到原来的 docs_with_score 中查找同一个文档，并取回它原来的向量分数。
            ranked_docs_with_score = [
                (doc, next(score for doc_, score in docs_with_score if doc_ == doc))
                for doc in ranked_docs
            ]
        else:#如果不想做重排序
            ranked_docs_with_score = docs_with_score

        # 3. 决定返回格式 这里的 true 和 False 是由调用者决定的（默认是false）--》
        if return_score:#如果return_score=True：[(Document, score),(Document, score)]（包含分数）
            return ranked_docs_with_score
        else:#return_score=False：[ Document,Document]
            return [doc for doc, _ in ranked_docs_with_score]

    def get_compresstion_retriever(self, search_kwargs: dict):#创建一个符合 LangChain 接口的检索器对象，用来在 RAG 链执行时完成“向量召回 + 可选重排序”。组装并返回一个检索器
        """
        获取适配Langchain链的检索器
        """
        search_kwargs = search_kwargs or {"k": settings.SEARCH_TOP_K}#默认5
        # 创建基础向量检索器：调用嵌入模型生成查询向量--》在 Chroma/FAISS 中查找相似向量--〉返回对应的 Document 列表
        base_retriever = self.vector_store._store.as_retriever(
            search_kwargs=search_kwargs
        )
        # 如果启用了Reranker，则创建压缩检索器
        if self.reranker:
            return ContextualCompressionRetriever(#ContextualCompressionRetriever 是一个包装器，它的工作过程是：收到 query｜调用 base_retriever 进行向量检索｜得到候选文档｜调用 base_compressor 进行重排序或压缩｜返回处理后的文档
                base_compressor=self.reranker,
                base_retriever=base_retriever
            )#最终执行流程可以理解为：Chroma 向量检索得到候选文档、CrossEncoder 计算 query 与每个文档的相关性、按相关性排序、返回 top_k 文档
        else:
            return base_retriever#如果没有启用 reranker此时直接返回普通向量检索器 Chroma 向量检索、直接返回结果
#所以：USE_RERANKER = True返回：ContextualCompressionRetriever( 向量检索 + CrossEncoder 重排序 ）
#USE_RERANKER = False 返回：普通向量检索器


# 单例实例（全局唯一，避免重复加载）
_rag_retriever_instance = None

def get_rag_retriever() -> RAGRetriever:
    """获取全局唯一的RAG检索器实例"""
    global _rag_retriever_instance
    if _rag_retriever_instance is None:
        _rag_retriever_instance = RAGRetriever()
    return _rag_retriever_instance
