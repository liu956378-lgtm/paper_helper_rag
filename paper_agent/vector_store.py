import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
class KnowledgeBase:
    def __init__(self, model_name="BAAI/bge-m3", device="cpu"):
        """
        初始化向量知识库
        :param model_name: 选用的 Embedding 模型路径或名称
        :param device: 运行设备，"cpu" 或 "cuda"
        """
        print(f"[*] 正在加载 Embedding 模型: {model_name} ...")
        
        # 模型运行参数
        model_kwargs = {'device': device}
        
        # 编码参数：强制进行向量归一化
        encode_kwargs = {'normalize_embeddings': True} 
        
        # 加载 BGE 向量模型
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        self.vector_store = None
        self.db_path = "./faiss_index"
        print("[+] 模型加载完成！")

    def build_index(self, chunks):
        """
        接收切分好的文本块，构建并持久化 FAISS 索引
        """
        print(f"[*] 正在将 {len(chunks)} 个文本块向量化并构建索引...")
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)
        
        self.vector_store.save_local(self.db_path)
        print(f"[+] 索引构建完成，已保存至 {self.db_path} 目录。")

    def load_index(self):
        """
        从本地加载已构建的 FAISS 索引
        """
        if os.path.exists(self.db_path):
            print(f"[*] 正在从 {self.db_path} 加载本地索引...")
            self.vector_store = FAISS.load_local(
                self.db_path, 
                self.embeddings,
                allow_dangerous_deserialization=True  # 允许加载本地生成的 pickle 文件
            )
            print("[+] 本地索引加载成功！")
        else:
            print("[-] 未找到本地索引，请先构建。")

    def search(self, query: str, top_k: int = 3):
        """
        执行向量检索
        """
        if not self.vector_store:
            raise ValueError("向量库未初始化，请先加载或构建索引。")
        
        print(f"[*] 正在检索 Query: '{query}'")
        # 返回结果为 [(Document, score), ...]
        return self.vector_store.similarity_search_with_score(query, k=top_k)


def main():
    # 1. 模拟经过 PDF 解析和切分后得到的文本块 (Chunks)
    mock_chunks = [
        Document(
            page_content="在 1D 质谱数据分析中，采用具有一维卷积核的 ResNet-1D 模型可以有效提取谱图的局部峰形特征。",
            metadata={"source": "deep_learning_ms.pdf", "page": 2}
        ),
        Document(
            page_content="Squeeze-and-Excitation (SE) 模块通过显式地建模通道间的依赖关系，自适应地重新校准通道特征响应，能够显著提升重叠峰的识别准确率。",
            metadata={"source": "deep_learning_ms.pdf", "page": 5}
        ),
        Document(
            page_content="直接离子化质谱（DIMS）省去了色谱分离步骤，导致混合物的质谱图高度复杂，需要更强大的解卷积算法来处理 0.001 m/z 级别的质量容差。",
            metadata={"source": "ambient_ionization_review.pdf", "page": 12}
        ),
        Document(
            page_content="传统的数据库匹配方法（如 NIST 谱库）在处理共流出物时容易出现假阳性，构建共峰参考谱库是未来的解决思路之一。",
            metadata={"source": "spectral_library_construction.pdf", "page": 8}
        )
    ]

    # 2. 初始化知识库 (首次运行会自动从 HuggingFace 下载 BGE 模型)
    kb = KnowledgeBase(device="cpu")

    # 3. 构建索引并保存到本地
    kb.build_index(mock_chunks)

    # 4. 执行检索测试
    print("\n" + "="*40)
    test_query = "如何利用注意力机制或者通道特征来处理重叠的化学信号？"
    
    # 我们期望它能精准召回关于 "SE 模块" 的那个 Chunk
    results = kb.search(test_query, top_k=2)

    # 5. 打印检索结果
    print(f"\n查询问题: {test_query}")
    for idx, (doc, score) in enumerate(results):
        print(f"\nTop {idx + 1} (相似度得分: {score:.4f}):")
        print(f"来源: {doc.metadata.get('source')} - 第 {doc.metadata.get('page')} 页")
        print(f"内容: {doc.page_content}")

if __name__ == "__main__":
    main()