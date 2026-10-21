import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

# ==========================================
# 1. 路径与配置设置
# ==========================================
# 统一路径格式，避免转义字符问题
cache_path = os.path.expanduser(r"C:\Users\1\.cache\huggingface\hub")
model_local_path = os.path.join(cache_path, "models--BAAI--bge-m3", "snapshots", "5617a9f61b028005a4858fdac845db406aefb181")
db_path = "./test_faiss_index"

# ==========================================
# 2. 文档加载与切分
# ==========================================
loader = TextLoader("./test.txt", encoding="utf-8")
documents = loader.load()

# ⚠️ 优化点：调整切分大小，保留完整的上下文语义
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,    # 建议 300-500 字符
    chunk_overlap=0,  # 增加 50 个字符的重叠，防止关键句子被从中间硬切断
    separators=["\n\n", "\n", "。", ".", " ", "?", "!"]
)
splits = text_splitter.split_documents(documents)

# ==========================================
# 3. 向量化模型加载
# ==========================================
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # 开启归一化计算余弦相似度更准

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_local_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# ==========================================
# 4. 向量库构建与更新
# ==========================================
if os.path.exists(db_path):
    print(f"[*] 发现已有索引，直接加载: {db_path}")
    vector_store = FAISS.load_local(
        db_path, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    # ⚠️ 删除了这里的 add_documents 和 save_local
    print("[+] 索引加载完毕，准备检索！")
else:
    print("[*] 未找到本地索引，正在创建新索引...")
    vector_store = FAISS.from_documents(splits, embeddings)
    vector_store.save_local(db_path)
    print("[+] 新索引已保存至磁盘，加载完毕！")
# 保存至磁盘，供下次复用
vector_store.save_local(db_path)
print("[+] 索引已保存至磁盘，加载完毕！")

# ==========================================
# 5. 语义检索测试
# ==========================================
query = '小刘的父亲是谁？'

# 检索前检查 (防御性编程)
if not vector_store:
    raise ValueError("向量库未初始化，请检查上方逻辑。")

results = vector_store.similarity_search_with_score(query, k=3)

print(f"\n📚 关于查询「{query}」的检索结果：")
for i, (doc, score) in enumerate(results):
    print(f"\n【结果 {i+1}】相似度分数: {score:.4f} (L2 距离，越小越相关)")
    # ⚠️ 优化点：增加切片展示长度，以便观察完整的句子
    print(f"内容: {doc.page_content[:10]}...") 
    if doc.metadata:
        print(f"元数据: {doc.metadata}")