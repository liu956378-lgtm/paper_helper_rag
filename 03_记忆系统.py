from langchain_community.llms import tongyi
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi
from typer import prompt

key='sk-2c7d93a45c384a389d70bf757d8d5a27'

chat=ChatTongyi(api_key=key,model_name="qwen3.7-max")

###  ConversationBufferWindowMemory固定上下文窗口;ConversationTokenBufferMemory按token数固定上下文窗口
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory,ConversationTokenBufferMemory ###  ConversationBufferWindowMemory固定上下文窗口
from langchain.chains import ConversationChain

##保留k轮对话，保留最新对话
memory=ConversationBufferWindowMemory(k=1)
conversation=ConversationChain(memory=memory, llm=chat)
conversation.predict(input="你好,我是程序猿")
conversation.predict(input="你是谁")

##查询记忆系统
print(memory.buffer)


##追加记忆
memory.save_context({"input":"Hi"},("output":"what's up"})
memory.chat_memory.add_user_message("你好")memory.chat_memory.add_ai_message("您好")
memory.save_context({"input":"Hi"},("output":"what's up"})

# ##现代可用
# # 1. 导入新组件（替换了旧导入）
# from langchain_community.chat_models import ChatTongyi
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory

# # 你的 key（建议从环境变量读取，更安全）
# key = "your-api-key-here"  # 请替换为你的真实 key

# # 2. 初始化模型（与你原来一致）
# chat = ChatTongyi(api_key=key, model_name="qwen3.7-max")

# # 3. 定义提示词模板（替代 ConversationChain 的内置模板）
# #    这里明确告诉模型：历史消息用 {history} 占位，用户新输入用 {input}
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是一个乐于助人的助手。"),  # 系统角色设定，可自定义
#     MessagesPlaceholder(variable_name="history"),  # 历史消息的占位符
#     ("human", "{input}")  # 用户当前输入
# ])

# # 4. 构建 LCEL 链（替代 ConversationChain 对象）
# #    这一步只是“组装”了提示词和模型，还没有记忆功能
# chain = prompt | chat

# # 5. 配置记忆存储（替代 ConversationBufferMemory）
# #    这是一个简单的内存存储，生产环境可替换为 Redis 等
# store = {}

# def get_session_history(session_id: str):
#     """根据 session_id 获取或创建对应的历史记录"""
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

# # 6. 创建带记忆的对话链（等效于 conversation = ConversationChain(...)）
# #    这是最关键的一步，用 RunnableWithMessageHistory 包装了上面的 chain
# conversation_with_memory = RunnableWithMessageHistory(
#     runnable=chain,
#     get_session_history=get_session_history,
#     input_messages_key="input",      # 指定输入字典中哪个 key 是用户问题
#     history_messages_key="history",  # 指定与模板中占位符对应的 key
# )

# # 7. 使用（等效于 conversation.predict）
# #    注意：现在每次调用都需要通过 config 指定 session_id 来区分不同会话
# session_config = {"configurable": {"session_id": "user_123"}}  # 可自定义 ID

# # 第一轮对话
# response1 = conversation_with_memory.invoke(
#     {"input": "你好，我是程序猿"},
#     config=session_config
# )
# print("助手:", response1.content)  # 注意返回的是 AIMessage 对象，用 .content 取文本

# # 第二轮对话（自动带上历史记忆）
# response2 = conversation_with_memory.invoke(
#     {"input": "你是谁"},
#     config=session_config
# )
# print("助手:", response2.content)