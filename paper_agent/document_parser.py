import fitz  # PyMuPDF 的引入名称
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class PDFParser:
    def __init__(self, chunk_size=800, chunk_overlap=150):
        """
        初始化解析器
        chunk_size: 每个文本块的最大字符数
        chunk_overlap: 相邻文本块的重叠字符数，用于保持上下文语义连贯
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 递归字符切分器：会按列表中分隔符的顺序依次尝试切分，直到满足 chunk_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # 分隔符优先级：段落 -> 换行 -> 中文句号 -> 英文句号 -> 空格 -> 单个字符
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )

    def parse_pdf(self, file_path: str, filename: str) -> list[Document]:
        """读取 PDF 并按页提取文本与元数据"""
        documents = []
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                
                # 数据清洗：去除空页、多余空格和不可见字符
                text = text.replace('\x00', '').strip()
                if not text:
                    continue
                
                # 核心操作：绑定 Metadata
                metadata = {
                    "source": filename,
                    "page": page_num + 1  # 页码从1开始计算，方便人类阅读
                }
                documents.append(Document(page_content=text, metadata=metadata))
                
            doc.close()
            return documents
        except Exception as e:
            print(f"解析 PDF {filename} 时发生错误: {e}")
            return []

    def parse_and_split(self, file_path: str, filename: str) -> list[Document]:
        """完整流程：读取 -> 清洗 -> 切分"""
        print(f"开始处理文献: {filename}")
        
        # 1. 提取整页文档
        raw_documents = self.parse_pdf(file_path, filename)
        if not raw_documents:
            return []
            
        # 2. 进行文本切分
        chunks = self.text_splitter.split_documents(raw_documents)
        
        print(f"处理完成！共生成 {len(chunks)} 个文本块。")
        return chunks

    
def run_test():
    # 1. 实例化解析器
    # 针对学术论文，800个字符大概对应半个双栏页面的文字量
    parser = PDFParser(chunk_size=800, chunk_overlap=150)
    
    # 2. 执行解析与切分
    pdf_path = r"C:\Users\1\Desktop\文献\研一下\dl\Applications of deep learning in metabolomics.pdf" # 请确保当前目录下有这个文件
    chunks = parser.parse_and_split(pdf_path, "Applications of deep learning in metabolomics.pdf")
    
    # 3. 打印结果进行抽样检查
    if chunks:
        print("\n--- 随机抽样检查 ---")
        
        # 打印第一个 Chunk
        print("\n[第一个文本块 (Chunk 1)]")
        print(f"元数据: {chunks[0].metadata}")
        print(f"内容长度: {len(chunks[0].page_content)} 字符")
        print(f"内容预览: {chunks[0].page_content[:200]}...\n")
        
        # 如果文本块数量较多，打印中间的一个 Chunk 检查连贯性
        mid_idx = len(chunks) // 2
        print(f"\n[中间文本块 (Chunk {mid_idx+1})]")
        print(f"元数据: {chunks[mid_idx].metadata}")
        print(f"内容预览:\n{chunks[mid_idx].page_content[:300]}")
        
        # 验证重叠 (Overlap) 是否生效
        if mid_idx > 0:
            print("\n--- 验证 Overlap ---")
            prev_chunk_tail = chunks[mid_idx - 1].page_content[-50:]
            curr_chunk_head = chunks[mid_idx].page_content[:50]
            print(f"上一个 Chunk 的尾部: {prev_chunk_tail}")
            print(f"当前 Chunk 的头部  : {curr_chunk_head}")
            # 如果配置生效，你能用肉眼看出头部和尾部有重复的句子

if __name__ == "__main__":
    run_test()