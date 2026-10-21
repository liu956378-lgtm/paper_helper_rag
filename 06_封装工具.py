from langchain.tools import tool


@tool
def add(a:int,b:int) -> int:
    """两个整数相加"""##大模型根据解释调用工具
    return a+b
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langsmith import Client
from langchain_community.chat_models import ChatTongyi

key='sk-2c7d93a45c384a389d70bf757d8d5a27'
llm=ChatTongyi(api_key=key,model_name="qwen-plus")

@tool
def sult(a:int,b:int) -> int:
    """两个整数相减"""
    return a-b

tools = [add,sult]

# 获取提示词
client = Client()

# 直接通过 Client 拉取并传入安全放行参数
prompt = client.pull_prompt("hwchase17/react", dangerously_pull_public_prompt=True)

#·创建AgentExecutor，运行智能体
agent = create_tool_calling_agent(llm=llm,tools=tools,prompt=prompt)
#·verbose代表输出日志

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
response = agent_executor.invoke({'input':'200加1223等于多少?'})
print(response)


##曲线救国：大模型调用工具时，把握不好参数数量，将文本传给工具函数，工具函数利用正则化函数再解析文本，提取参数
import re
@tool
def multiply(input_str:str) ->int:
    """计算两个整数的乘积，输入格式为a*b
args:
  input_str:输入的字符串，格式为a*b
returns: int
"""
    match = re.findall(r'\d+',input_str)
    if len(match) == 2:
    a,b = map(int,match)  
    #map函数将列表中的元素依次传递给int，转换int类型 

    return a * b



# ##调用api
# import requests
# def get_weather(city: str) -> str:
#     """获取指定城市的天气信息"""
#     api_key = "your_api_key"  # 替换为你的实际API Key
#     url = f"http
#     import requests
#     def get_weather(city:str) ->str:
#     调用实时天气API，返回温度及天气状况
#     city:城市名称，如:"北京
#     payloadheaders = 
#     url = "https://eolink.o.apispace.com/456456/weather/ve01/now
#     {"areacode" : "101010100"}
#     "X-APISpace-Token":"h72uhbhva2t9s4k5ozrx23f8vmwqcdse"
#     response=requests.get(url, params=payload, headers=headers)
#     return response.text

# get_weather('北京')

#pandas读取csv文件
# city_df = pd.read_csv("city.csv")
# city_df['district'] == city_name：创建布尔序列，标记匹配的行
# city_df[条件]：筛选出所有符合条件的行