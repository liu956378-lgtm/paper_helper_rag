from langchain.memory import ConversationSummaryBufferMemory

from langchain.chains import ConversationChain

from langchain_community.chat_models import ChatTongyi

key='sk-2c7d93a45c384a389d70bf757d8d5a27'
llm=ChatTongyi(api_key=key,model_name="qwen-plus")




# 2. 创建摘要缓冲内存
# max_token_limit: 触发摘要的 token 阈值
# 当缓冲区的 token 数超过此值，旧消息会被压缩成摘要
memory = ConversationSummaryBufferMemory(
    llm=llm,                     # 用于生成摘要的模型
    max_token_limit=100,         # 阈值：超过100 tokens 触发压缩
    return_messages=True,        # 返回消息列表格式
)

# 3. 创建对话链
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,                # 查看详细日志
)

# 4. 模拟对话
# 第一次对话（内容较少，未触发压缩）
response1 = conversation.predict(input="我叫小明，是一名软件工程师")
print(f"回答1: {response1}\n")

# 第二次对话
response2 = conversation.predict(input="我最喜欢的编程语言是Python")
print(f"回答2: {response2}\n")

# 第三次对话（累积内容可能已超阈值，触发摘要）
response3 = conversation.predict(input="我最近在学习机器学习")
print(f"回答3: {response3}\n")

# 5. 查看压缩后的内存内容
print("=" * 50)
print("当前内存内容:")
print(memory.buffer)  # 可能包含摘要和最近几条消息