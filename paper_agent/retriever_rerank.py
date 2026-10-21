from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-base", device="cuda"):
        # Reranker 模型通过 Cross-Attention 比较 query 和 doc 的真实语义相关度
        self.reranker = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, retrieved_docs: list, top_k: int = 3):
        """对召回的文档进行交叉打分精排"""
        if not retrieved_docs:
            return []

        # 组装 (query, doc_text) 对
        pairs = [[query, doc.page_content] for doc, score in retrieved_docs]
        
        # 预测相关性得分
        scores = self.reranker.predict(pairs)
        
        # 将得分与原文档绑定并排序
        doc_score_pairs = list(zip(retrieved_docs, scores))
        # 按 Rerank 降序排列
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        
        # 返回精排后的 Top-K 文档（提取原 Document 对象）
        reranked_docs = [item[0][0] for item in doc_score_pairs[:top_k]]
        return reranked_docs

# --- 代码解释 ---
# 面试亮点：这是简历中 "相关性排序(Rerank)" 的直接体现。
# 双塔模型(Embedding)召回Top-15，然后用精排模型打分提取Top-3喂给大模型。
# 极大降低了大模型处理不相关上下文时的“幻觉”概率。