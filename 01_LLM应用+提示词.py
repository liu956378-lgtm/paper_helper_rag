from langchain_community.llms import tongyi
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi

key='sk-2c7d93a45c384a389d70bf757d8d5a27'

##创建对象，实例化类
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个精通{domain}领域的资深专家，回答要尽量简明扼要。"),
    ("human", "请帮我解释一下：{question}") 
])

llm=ChatTongyi(api_key=key,model_name="qwen3.7-max")


chain = prompt | llm  ##现代可用；prompt | llm | output_parser；|：管道 
chunk=[]
##流式输出stream
for c in chain.stream({'domain':'ai','question':'现在你是什么模型，具体模型名字是什么？'}):
    chunk.append(c)
    print(c.content,end='',flush=True)
    


