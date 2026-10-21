from langchain_community.llms import tongyi
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi

key='sk-dfcb61eed9a34583a9323d8ef10012af'

##LLM的"自由文本"变成"程序可用的结构化数据
from langchain.output_parsers import StructuredOutputParser,ResponseSchema,PydanticOutputParser

name_Response = ResponseSchema(name="name", description="人的名字",type="array")
age_Response = ResponseSchema(name="age", description="人的年龄",type="interger")

response_schemas = [name_Response, age_Response]    

##创建输出解析器;  取决于解析器的定义，通常是 字典（Dict）、列表（List） 或自定义的 Pydantic 对象（Object
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

##创建提示模板，包含格式化命令
template="""你是一个信息提取助手，请从以下文本中提取姓名和年龄，并以JSON 格式返回:
文本:{input_text}
{format_instructions}"""

prompt = PromptTemplate(
template=template,
partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

llm=ChatTongyi(api_key=key,model_name="qwen3.7-max")
input_text="小明今年20岁了，他的朋友小红今年18岁。"
filled_prompt = prompt.format(input_text=input_text)
resp=llm.invoke(filled_prompt)
parsed_output = output_parser.parse(resp.content)
print(parsed_output)
