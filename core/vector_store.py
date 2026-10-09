"""
向量库管理模块——封装chroma或者faiss操作，提供统一接口
通过embedding模块获取嵌入模型，转换为向量后，使用chroma或者faiss存储到本地向量库持久化存储，方便后续的检索与查询等操作。
"""
import logging
import shutil
from langchain.schema import Document
from langchain_community.vectorstores import Chroma, FAISS
from config.settings import settings
from core.embedding import get_embedding_model

logger = logging.getLogger(__name__)#记录日志

class VectorStoreManager:
    """向量库管理器"""
    def __init__(self):
        self.embedding_model = get_embedding_model()#初始化向量模型
        self._store = None#私有属性store 默认为None
        self._initialize_vector_store()#自动初始化 向量数据库方法

    def _initialize_vector_store(self):#初始化向量数据库
        """初始化向量库"""
        if settings.VECTOR_STORE_TYPE == "chroma":#如果配置的向量数据库是chroma 就初始化这个数据库
            self._store = Chroma(
                persist_directory=str(settings.VECTOR_DB_DIR),
                embedding_function=self.embedding_model
            )
            logger.info(f"Chroma向量库已初始化，存储路径: {settings.VECTOR_DB_DIR}")#记录日志
        elif settings.VECTOR_STORE_TYPE == "faiss":#如果配置的向量数据库是faiss 就初始化faiss这个数据库
            faiss_index_path = settings.VECTOR_DB_DIR / "index.faiss"
            if faiss_index_path.exists():#首先检查 FAISS 索引文件是否存在 如果存在，就加载原来的索引
                self._store = FAISS.load_local(
                    str(settings.VECTOR_DB_DIR),
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"FAISS向量库已加载，存储路径: {settings.VECTOR_DB_DIR}")#记录日志
            else:#如果索引不存在，就创建一个新的 FAISS 向量库
                self._store = FAISS.from_documents(
                    [Document(page_content="初始化向量库", metadata={"source": "init"})],
                    self.embedding_model
                )
                self._save_faiss()
                logger.info(f"FAISS向量库已保存，存储路径: {settings.VECTOR_DB_DIR}")
        else:
            raise ValueError(f"不支持的向量库类型: {settings.VECTOR_STORE_TYPE}")#否则不支持 只支持chroma 和 faiss这两个向量数据库

    def _save_faiss(self):#保存faiss数据库的方法 Chroma 和 FAISS 的持久化机制不同 Chroma 自带持久化机制FAISS 主要是内存索引，需要手动保存
        """保存FAISS向量库"""
        if settings.VECTOR_STORE_TYPE == "faiss":
            self._store.save_local(str(settings.VECTOR_DB_DIR))

    def add_documents(self, documents: list) -> int:#添加文档到向量数据库中
        """
        添加文档到向量库
        :param documents: 文档列表
        :return: 添加的文档数量
        """
        if not documents:#如果没有文档列表 返回0 表示添加了0个文档
            return 0
        try:
            self._store.add_documents(documents)#传入文档列表 并添加到向量数据库中
            if settings.VECTOR_STORE_TYPE == "faiss":
                self._save_faiss()#如果向量数据库是faiss则需要手动保存一下
            logger.info(f"成功添加 {len(documents)} 个文档到向量库")#记录日志
            return len(documents)#返回成功添加的数量
        except Exception as e:
            logger.error(f"添加文档到向量库失败: {e}")#出现异常记录日志 返回0
            return 0

    def delete_by_source(self, source: str) -> int:#按源文件路径删除文档片段的统一入口 删除向量数据库中所有 metadata["source"] 等于指定路径的文本片段。
        """
        根据源文件删除文档
        :param source: 源文件路径
        :return: 是否删除成功
        """
        if settings.VECTOR_STORE_TYPE == "chroma":#如果是chroma
            results = self._store.get(where={"source": source})#先根据元数据查询匹配的文档
            ids_to_delete = results.get("ids", [])#接着取出 ID 如果查询结果中没有 ids，就返回空列表
            if ids_to_delete:#如果找到了记录 它会一次性删除所有匹配的向量片段。
                self._store.delete(ids=ids_to_delete)
                logger.info(f"成功删除源文件' {source} '的 {len(ids_to_delete)} 个片段")#记录日志
                return len(ids_to_delete)#返回删除的数量
            return 0
        else:
            logger.warning(f"FAISS模式下删除功能比较慢，建议使用Chroma")#记录警告 FAISS 分支
            return self._delete_by_source_faiss(source)
#Chroma 删除：按 metadata 过滤 → 批量删除 → 返回删除片段数
#FAISS 删除：遍历文档 → 重建索引 → 保存索引 → 返回删除片段数

    def _delete_by_source_faiss(self, source: str) -> int:#faiss模式下的删除 思路：遍历 FAISS 中的所有 Document
#→ 找到 source 匹配的片段并统计
#→ 保留 source 不匹配的片段
#→ 用保留的片段重新创建 FAISS 索引
#→ 保存新索引
        """
        FAISS模式的删除，是需要重构索引的
        """
        doc_ids = list(self._store.docstore._dict.keys())
        to_keep = []
        deleted_count = 0
        for doc_id in doc_ids:
            doc = self._store.docstore.search(doc_id)
            if doc.metadata.get("source") == source:
                deleted_count += 1
            else:
                to_keep.append(doc)
        if deleted_count > 0:
            # 重构索引
            self._store = FAISS.from_documents(
                to_keep,
                self.embedding_model
            )
            self._save_faiss()
            logger.info(f"成功删除源文件' {source} '的 {deleted_count} 个片段")
        return deleted_count

#相似度检索 是向量库的纯相似度检索接口，它接收一个问题，在向量库中查找最相似的前 k 个文档片段，并返回 Document 列表。
    def similarity_search(self, query: str, k: int = None, filter_dict: dict = None) -> list:
        """
    在向量库中执行纯相似度检索。

    :param query: 查询文本
    :param k: 返回的文档数量，未传时使用 settings.SEARCH_TOP_K
    :param filter_dict: Chroma 元数据过滤条件
    :return: 最相关的 Document 列表
    """
        k = k or settings.SEARCH_TOP_K#k 默认是5
        if filter_dict and settings.VECTOR_STORE_TYPE == "chroma":# 传入过滤条件且当前使用 Chroma 时，按元数据过滤检索
            return self._store.similarity_search(query, k=k, filter=filter_dict)
        else:
            return self._store.similarity_search(query, k=k)# 没有过滤条件，或当前使用 FAISS 时，执行普通向量检索

#在向量库中执行相似度检索，返回最相似的前 k 个文档，同时返回每个文档对应的向量检索分数。 他和   similarity_search  的区别是 一个只返回文档 一个返回“文档 + 分数”
    def similarity_search_with_score(self, query: str, k: int = None, filter_dict: dict = None) -> list:
        """ 相似度检索并返回相关性分数 """
        k = k or settings.SEARCH_TOP_K
        if filter_dict and settings.VECTOR_STORE_TYPE == "chroma":
            return self._store.similarity_search_with_score(query, k=k, filter=filter_dict)
        else:
            return self._store.similarity_search_with_score(query, k=k)

#返回当前向量库的统计信息，主要是向量库中有多少个文档片段，以及当前使用的是哪种向量库、数据保存在哪里。
    def get_collection_stats(self) -> dict:
        """ 获取向量库统计信息 """
        if settings.VECTOR_STORE_TYPE == "chroma":
            count =  self._store._collection.count()
            return {
                "total_documents": count,
                "vector_store_type": "chroma",
                "persist_directory": str(settings.VECTOR_DB_DIR)
            }
        else:
            count = self._store.index.ntotal
            return {
                "total_documents": count,
                "vector_store_type": "faiss",
                "persist_directory": str(settings.VECTOR_DB_DIR)
            }
#清空向量数据库
    def clear_all(self):
        """ 清空向量库 """
        if settings.VECTOR_STORE_TYPE == "chroma":
            self._store.delete_collection()
            self._initialize_vector_store()
        else:
            shutil.rmtree(settings.VECTOR_DB_DIR, ignore_errors=True)
            settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
            self._initialize_vector_store()
        logger.info("向量库已清空")

# 全局单例
_vector_store_instance: VectorStoreManager = None

def get_vector_store_manager() -> VectorStoreManager:
    """ 获取向量库管理器单例 """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreManager()
    return _vector_store_instance
