from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os   

cache_path = os.path.expanduser(r"C:\Users/1/.cache/huggingface/hub/")
model_local_path = os.path.join(cache_path, "models--BAAI--bge-m3", "snapshots", "5617a9f61b028005a4858fdac845db406aefb181")
embeddings = HuggingFaceBgeEmbeddings(model_name=model_local_path,model_kwargs={"device": "cpu"},encode_kwargs={"normalize_embeddings":True})
query = "请帮我解释一下：这个产品太垃圾了"
vector = embeddings.embed_query(query)
print(vector)