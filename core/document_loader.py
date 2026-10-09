"""
文档加载器模块——支持 pdf word excel txt markdown 等多种格式的文档加载
步骤：
1. 加载文档：使用合适的加载器，可以选择是否添加元数据。
    什么是元数据？——就是描述数据的数据，例如：文件名、文件类型、作者等等等
    添加元数据，优点就是能更好的管理数据。
2. 分割文档：将文档分割成多个片段，以便于后续的处理。
"""
from pathlib import Path
from config.settings import settings
import logging
logger = logging.getLogger(__name__)
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentLoader:#文档加载类
    """统一文档加载器类"""
    # 支持的文档扩展名映射 都加载进来 pdf docx doc txt md xlsx xls
    LOADER_MAP= {
        ".pdf": PyMuPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader
    }

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):#设立对象初始化的属性 切块大小、切块覆盖度、文本分割器（。。、。。、递归切割的方法、用 Python 内置的 len() 来计算文本长度，并据此判断一个文本块是否达到 chunk_size）
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            length_function=len
        )

    def load_file(self, file_path: str, metadata: dict = None) -> list:#加载单个文件的方法 作用是：接收一个文件路径，选择正确的 LangChain Loader 读取文件，附加元数据，切分成文档片段，最后返回这些片段。
        """
        加载单个文件
        :param file_path: 文件路径
        :param metadata: 可选的元数据字典
        :return: 文档片段列表
        """
        # 1. 先检查文件是否存在
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return []
        # 2. 检查文件类型
        ext = file_path.suffix.lower()
        if ext not in self.LOADER_MAP:
            logger.error(f"不支持的文件类型: {ext}")
            return []
        try:
            # 3. 加载文档
            loader_class = self.LOADER_MAP[ext]
            if ext == ".txt":
                loader = loader_class(str(file_path), encoding="utf-8")
            else:
                loader = loader_class(str(file_path))
            documents = loader.load()
            logger.info(f"成功加载文件: {file_path}， 原始文档书段数: {len(documents)}")
            # 4. base_metadata添加自定义元数据 字段包括：来源、文件名、文件类型（扩展名）
            base_metadata= {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": ext[1:],       # 过滤 .txt 的 . 保留 txt
            }
            if metadata:#如果调用 load_file() 时传入了额外元数据，就把它合并到 base_metadata。
                base_metadata.update(metadata)
            for doc in documents:
                doc.metadata.update(base_metadata)
            # 5. 分割文档
            split_docs = self.text_splitter.split_documents(documents)#split——docs分割后的 Document 列表
            logger.info(f"成功分割文档: {file_path}， 分割后的文档数: {len(split_docs)}")#日志：记录成功分割的文档是哪个，分了多少块
            return split_docs#返回被分好的文本块
        except Exception as e:
            logger.error(f"加载文件失败: {file_path}, 错误: {e}")#记录失败的日志
            return []#返回一个空列表

    def load_directory(self, dir_path: str, recursive: bool = True) -> list:#批量加载目录中文件的入口方法。它遍历目录下的文件，逐个调用 load_file()，最后把所有文件切分后的文档片段合并成一个列表返回。
        """
        加载整个目录下的所有文件
        :param dir_path: 目录路径
        :param recursive: 是否递归加载子目录
        :return: 文档片段列表
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.error(f"目录不存在: {dir_path}")#记录日志 文件不存在 返回一个空列表
            return []
        # 遍历目录下的所有文件
        all_docs = []#初始化all_docs空列表
        patten = "**/*" if recursive else "*"
        for file_path in dir_path.rglob(patten):#开始遍历 rglob() 会根据模式递归查找路径，每次循环得到的 file_path 是一个 Path 对象
            if file_path.is_file() and file_path.suffix.lower() in self.LOADER_MAP:#文件扩展名在支持列表中
                docs = self.load_file(str(file_path))#加载文件 load_file() 返回的不是原始文件内容，而是已经切分好的文档片段列表
                all_docs.extend(docs)#把它们添加到之前初始化的all_docs空列表中
        logger.info(f"成功加载目录: {dir_path}, 共 {len(all_docs)} 个文档片段")#记录成功的日志信息
        return all_docs#返回 目录中所有受支持文件经过加载、添加元数据并切分后得到的全部 Document 片段列表
