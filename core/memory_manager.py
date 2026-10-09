"""
多轮对话记忆管理模块——支持多会话隔离
"""
import uuid
import logging
from datetime import datetime, timedelta
from langchain.memory import ConversationBufferWindowMemory

logger = logging.getLogger(__name__)#大模型记录日志
#单记忆会话管理
class ConversationSession:
    """ 单个会话的记忆管理 """
    def __init__(self, session_id: str, window_size: int = 5):
        self.session_id = session_id#会话ID
        self.create_at = datetime.now()#创建此会话的时间
        self.last_active = datetime.now()#最后一次使用此会话的时间
        self.memory = ConversationBufferWindowMemory(#记忆使用的是滑动记忆窗口
            k=window_size,  # k参数是指窗口大小 即最多保留多少条对话记录
            memory_key="chat_history",  # memory_key是指对话历史的键名
            return_messages=True,   # return_messages是指是否返回消息列表
            output_key="answer" # output_key是指输出的键名
        )
        self.metadata: dict = {}#初始化一个保存元数据的字典

    def add_user_message(self, message: str):#添加用户的消息 记录用户的输入 和最后一次输入的时间
        """ 添加用户消息 """
        self.memory.chat_memory.add_user_message(message)
        self.last_active = datetime.now()

    def add_ai_message(self, message: str):#添加ai消息 记录ai的输出 和最后一次输入的时间
        """ 添加AI消息 """
        self.memory.chat_memory.add_ai_message(message)
        self.last_active = datetime.now()

    def get_memory(self):#获取当前会话记忆的内容
        """ 获取当前会话的记忆 """
        return self.memory.chat_memory.messages

    def clear(self):#清空当前会话记忆的内容 和最后一次时间
        """ 清空当前会话的记忆 """
        self.memory.clear()
        self.last_active = datetime.now()

    def to_dict(self):#将信息转换为字典格式的方法 追加角色 和 内容 ；并返回五个信息
        """ 转换为字典 """
        messages = []
        for msg in self.memory.chat_memory.messages:
            messages.append({
                "role": msg.type,
                "content": msg.content
            })
        return {
            "session_id": self.session_id,
            "create_at": self.create_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "messages": messages,
            "metadata": self.metadata
        }
#多个会话的管理
class MemoryManager:
    """ 多会话记忆管理 """
    def __init__(self, window_size: int = 5, session_ttl_minutes: int = 60):
        self.sessions: dict = {}#字典记录多个session会话id
        self.window_size = window_size#窗口大小 默认是5
        self.session_ttl = timedelta(minutes=session_ttl_minutes)#
        logger.info(f"记忆管理器初始化 window_size={window_size}, session_ttl={session_ttl_minutes} minutes")#记录日志

    def create_session(self, session_id: str):#创建新会话
        """ 创建新会话 """
        if session_id is None:
            # uuid4() 是一个全局唯一标识符 当然也可以使用时间戳
            session_id = str(uuid.uuid4())#如果之前没有会话id就用uuid4创建一个
        if session_id in self.sessions:
            logger.warning(f"会话 {session_id} 已存在，将返回现有会话")
            return session_id#如果已经存在就返回现有的会话
        session = ConversationSession(session_id, self.window_size)#创建新的 ConversationSession 代表一次独立的多轮对话
        self.sessions[session_id] = session#把会话保存到字典
        logger.info(f"创建新会话: {session_id}")#记录日志
        return session_id#返回会话的id

    def get_session(self, session_id: str):#根据 ID 从字典中取出对应的会话对象
        """ 获取会话 """
        session = self.sessions.get(session_id)#将得到的会话ID赋值给session
        if session:
            session.last_active = datetime.now()# 如果会话对象存在，更新最后活跃时间
        return session# 返回 ConversationSession 对象

    def get_or_create_session(self, session_id: str):#MemoryManager 中常用的获取或创建会话方法
        """ 获取或创建会话 """
        #根据传入的 session_id 查找已有的 ConversationSession；如果找不到，就创建一个新会话，最后统一返回 ConversationSession 对象。
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        new_session_id = self.create_session(session_id)
        return self.sessions[new_session_id]

    def add_exchange(self, session_id: str, question: str, answer: str):#添加对话记录 格式包括：会话id 用户提问 ai的回答
        """ 添加一轮对话记录 """
        session = self.get_or_create_session(session_id)
        session.add_user_message(question)
        session.add_ai_message(answer)
        logger.info(f"添加对话记录: 会话 {session_id}, 用户: {question}, AI: {answer}")#记录日志

    def get_chat_history(self, session_id: str):#获取会话的聊天历史
        """ 获取会话的聊天历史 """
        session = self.get_session(session_id)#通过id找到对应的聊天历史
        if not session:
            return []#如果聊天历史不存在 返回一个空的列表
        return session.get_memory()#如果存在就返回对应的记忆

    def clear_session(self, session_id: str):#清空对应的会话
        """ 清空会话 """
        if session_id in self.sessions:#如果会话id存在
            self.sessions[session_id].clear()#根据对应的id执行clear方法
            logger.info(f"清空会话: {session_id}")#记录日志

    def delete_session(self, session_id: str):#删除会话
        """ 删除会话 """
        if session_id in self.sessions:#如果会话id存在
            del self.sessions[session_id]#删除对应的会话
            logger.info(f"删除会话: {session_id}")#记录日志

    def cleanup_old_sessions(self):#清理过期的会话 它根据每个会话的 last_active 与当前时间的差距，判断会话是否过期。默认过期时间是 60 分钟，在 MemoryManager.__init__() 中设置
        """ 清理过期会话 """
        now = datetime.now()#记录时间 如果会话空闲的时间超过60分钟 就清除
        expired_sessions = [#遍历 找出所有过期会话的 ID
            session_id for session_id, session in self.sessions.items()
            if now - session.last_active > self.session_ttl
        ]
        for session_id in expired_sessions:#将在expired_sessions列表当中的所有过期会话id删除
            self.delete_session(session_id)
        if expired_sessions:
            logger.info(f"清理过期会话: 已删除 {len(expired_sessions)} 个会话")#记录日志
        return len(expired_sessions)#返回删除数量

    def list_sessions(self):#列出当前所有的会话
        """ 列出所有会话 """
        return [session.to_dict() for session in self.sessions.values()]

    def get_default_memory(self):#获取默认会话记忆
        """ 获取默认会话的记忆 """
        default_session = self.get_or_create_session("default")
        return default_session.memory



# 全局单例
_memory_manager_instance = None

def get_memory_manager():#是 MemoryManager 的全局获取函数，作用是确保整个 Python 进程只创建一个记忆管理器，并把这个实例提供给所有需要保存聊天记录的模块使用。
    """ 获取以及管理器单例 """
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager()
    return _memory_manager_instance
