import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

# 导入你的业务逻辑模块
from document_parser import PDFParser
from vector_store import KnowledgeBase
from retriever_rerank import Reranker
from llm_generator import QAGenerator

# ================= 1. 全局资源初始化 =================
pdf_parser = PDFParser()
kb = KnowledgeBase(device="cpu") 
reranker = Reranker(device="cpu")

# 请替换为真实的 API Key。如果是本地测试，可以使用 Ollama 等本地模型接口
qa_engine = QAGenerator(
    model_name= "qwen-plus", 
    api_key= "sk-2c7d93a45c384a389d70bf757d8d5a27", 
    
    temperature = 0.1
)

# ================= 2. 生命周期管理 (现代化写法) =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动前执行：加载已有的 FAISS 向量库
    print("系统启动中：正在加载向量知识库...")
    kb.load_index()
    print("向量知识库加载完毕！")
    
    yield # 此时应用开始处理请求
    
    # 停止后执行：资源清理（例如关闭数据库连接）
    print("系统关闭：清理资源...")

# ================= 3. 创建 FastAPI 实例 =================
app = FastAPI(
    title="科研文献 RAG 问答系统",
    description="支持 PDF 上传解析、知识库构建与流式问答",
    version="1.0.0",
    lifespan=lifespan
)

# ================= 4. 路由定义 =================

@app.post("/api/v1/knowledge/upload", summary="上传 PDF 并构建知识库")
async def upload_pdf(file: UploadFile = File(...)):
    """接收用户上传的 PDF，进行解析、切片并入库。"""
    temp_path = f"temp_{file.filename}"
    try:
        # 将上传的文件流异步写入本地临时文件
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 1. 解析与切分 (耗时操作)
        chunks = pdf_parser.parse_and_split(temp_path, file.filename)
        
        # 2. 构建并保存向量库 (耗时操作)
        kb.build_index(chunks)
        
        return {
            "code": 200, 
            "status": "success", 
            "message": f"成功解析文件 '{file.filename}'，共生成 {len(chunks)} 个向量块并入库。"
        }
    except Exception as e:
        return {"code": 500, "status": "error", "message": f"处理失败: {str(e)}"}
    finally:
        # 确保清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/api/v1/chat/stream", summary="科研文献智能问答 (流式输出)")
async def query_rag_stream(question: str):
    """接收用户提问，检索文献并以流式打字机效果返回大模型生成的答案。"""
    
    # 1. 向量粗排召回 (召回 Top 15)
    retrieved_results = kb.search(question, top_k=15)
    
    # 2. Rerank 精排 (筛选出最相关的 Top 3)
    best_docs = reranker.rerank(question, retrieved_results, top_k=3)
    
    # 如果完全没有检索到任何相关内容，直接返回
    if not best_docs:
        # yield 的机制要求流式接口也必须返回迭代器
        async def empty_response():
            yield "对不起，知识库中未找到与您问题相关的内容。"
        return StreamingResponse(empty_response(), media_type="text/plain")

    # 3. 大模型生成回答 (返回生成器)
    return StreamingResponse(
        qa_engine.stream_generate(question, best_docs), 
        media_type="text/event-stream"
    )
