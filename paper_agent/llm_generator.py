import os
from typing import AsyncGenerator, List, Dict, Any, Tuple
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document

class QAGenerator:
    def __init__(
        self, 
        model_name: str = "qwen-plus", 
        api_key: str = "sk-2c7d93a45c384a389d70bf757d8d5a27", 
        temperature: float = 0.1
    ):
        """
        初始化大模型生成器
        :param model_name: 千问模型版本，常用 qwen-turbo (快/廉价), qwen-plus (均衡), qwen-max (强推理)
        :param api_key: 阿里 DashScope API Key，若为 None 则自动读取环境变量 DASHSCOPE_API_KEY
        :param temperature: 随机度，科研 RAG 建议设置为 0.0 - 0.2 以确保事实严谨
        """
        # 若传入了 api_key，存入环境变量，ChatTongyi 会自动读取
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key
            
        # 1. 实例化 LangChain 的 ChatTongyi 对象
        self.llm = ChatTongyi(
            model=model_name,
            temperature=temperature,
            streaming=True  # 开启流式输出模式
        )
        
        # 2. 构建符合科研 RAG 范式的 Prompt 模板
        self.prompt_template = self._init_prompt_template()

    def _init_prompt_template(self) -> ChatPromptTemplate:
        """配置系统级防幻觉 Prompt 与用户输入 Prompt"""
        system_template = (
            "你是一个严谨、专业的科研学术助手。\n"
            "你的任务是根据给定的【参考文献片段】，回答用户的学术问题。\n\n"
            "【回答规则】：\n"
            "1. 必须**严格基于**提供的【参考文献片段】进行回答，不要引入外部未经证实的信息。\n"
            "2. 如果参考片段中的内容不足以回答问题，请直接回答：“根据提供的文献内容，无法回答该问题。”，严禁凭空捏造。\n"
            "3. 回答逻辑要清晰、学术化，符合论文阅读习惯。\n\n"
            "【参考文献片段】：\n"
            "{context}"
        )
        
        human_template = "问题：{question}"

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    def format_context(self, docs: List[Document]) -> Tuple[str, List[str]]:
        """
        将精排后的 Document 列表转化为 Prompt 中的上下文字符串，并提取元数据
        """
        context_str = ""
        sources = []
        
        for idx, doc in enumerate(docs):
            source_file = doc.metadata.get("source", "未知文献")
            page_num = doc.metadata.get("page", "未知页码")
            source_tag = f"文献《{source_file}》 第 {page_num} 页"
            
            sources.append(source_tag)
            context_str += f"\n[片段 {idx + 1}] (出处: {source_tag}):\n{doc.page_content.strip()}\n"
            
        return context_str, sources

    async def stream_generate(self, question: str, docs: List[Document]) -> AsyncGenerator[str, None]:
        """
        异步流式生成答案（用于前端打字机效果及 FastAPI SSE 输出）
        """
        # 1. 格式化上下文与来源
        context_str, sources = self.format_context(docs)
        
        # 2. 组装 Prompt
        prompt_messages = self.prompt_template.format_messages(
            context=context_str,
            question=question
        )
        
        # 3. 调用 LangChain 的 astream 异步流式接口
        async for chunk in self.llm.astream(prompt_messages):
            # chunk 是 AIMessageChunk 对象，.content 为当前增量文本
            if chunk.content:
                yield chunk.content

        # 4. 答案生成完成后，末尾追加【文献来源溯源】
        yield "\n\n---\n**【文献来源溯源】**:\n"
        # 去重输出来源
        unique_sources = list(dict.fromkeys(sources))
        for idx, src in enumerate(unique_sources):
            yield f"{idx + 1}. {src}\n"


import asyncio
import os
from langchain_core.documents import Document
from llm_generator import QAGenerator

async def main():
    # 1. 检查环境变

    print(">>> 初始化 ChatTongyi 生成器...")
    qa_gen = QAGenerator(model_name="qwen-plus", temperature=0.1)

    # 2. 构造模拟检索结果 (Mock Documents)
    mock_docs = [
        Document(
            page_content="ResNet-1D 网络在一维质谱（Mass Spectrometry, MS）信号解析中具有显著优势。"
                         "通过引入残差连接，网络深度可扩展至34层，有效解决了重叠峰解卷过程中的梯度消失问题。",
            metadata={"source": "Analytical_Chemistry_2025.pdf", "page": 4}
        ),
        Document(
            page_content="配合 Squeeze-and-Excitation (SE) 注意力机制模块，ResNet-1D 可以自动增强高响应质谱特征峰的权重，"
                         "在 0.001 m/z 质量公差范围内，复合重叠峰的分辨率达到了 94.2%。",
            metadata={"source": "Analytical_Chemistry_2025.pdf", "page": 5}
        )
    ]

    # 3. 测试用例 1：有对应答案的查询
    query_1 = "ResNet-1D 在质谱重叠峰处理中的主要优势是什么？分辨率达到多少？"
    print(f"\n==================== 测试用例 1 ====================")
    print(f"用户问题: {query_1}")
    print("大模型回答 (流式输出中): \n")

    # 循环读取异步生成器
    async for chunk in qa_gen.stream_generate(query_1, mock_docs):
        # flush=True 保证终端立刻刷新打字效果
        print(chunk, end="", flush=True)
    print("\n===================================================\n")

    # 4. 测试用例 2：知识库中不存在的查询（验证防幻觉拒答能力）
    query_2 = "质谱仪的激光源需要多久更换一次？"
    print(f"==================== 测试用例 2 (防幻觉测试) ====================")
    print(f"用户问题: {query_2}")
    print("大模型回答 (预期应拒答): \n")

    async for chunk in qa_gen.stream_generate(query_2, mock_docs):
        print(chunk, end="", flush=True)
        
    print("\n===================================================")

if __name__ == "__main__":
    # 运行异步测试
    asyncio.run(main())