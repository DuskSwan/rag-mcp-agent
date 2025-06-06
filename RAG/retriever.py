from pathlib import Path
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# --- 全局配置 ---
# rag_dir = Path('RAG')
rag_dir = Path(__file__).parent
# 缓存文件，用于存储已处理的URL内容，避免重复抓取
CACHE_FILE = rag_dir / 'cache.json'
# 向量索引文件，用于存储FAISS索引
INDEX_FILE = rag_dir / 'index.faiss'
# sentence-transformer 模型，选择一个强大的多语言模型
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# --- 1. 数据加载与网页抓取 ---

def load_urls_from_file(file_path: str):
    """从指定的txt文件中加载URL列表。"""
    assert os.path.exists(file_path), f"文件 {file_path} 不存在。请检查路径是否正确。"
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def load_cache():
    """加载缓存文件。"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """保存缓存到文件。"""
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4)

def scrape_and_process_url(url_line):
    """
    抓取单个URL的内容，并提取文本。
    能处理 "url" 或 "url description" 格式的行。
    """
    parts = url_line.split()
    url = parts[0]
    description = ' '.join(parts[1:])

    print(f"正在处理: {url}")
    return f"{description}: {url}".strip()  # 简化处理，直接使用描述和URL

# --- 2. 检索器核心类 ---

class UrlRetriever:
    def __init__(self, model_name=MODEL_NAME):
        print("正在加载 Sentence Transformer 模型...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.url_list = []

    def build_index(self, url_lines, force_rebuild=False):
        """
        为URL列表构建或加载FAISS索引。
        
        参数:
        url_lines (list): 从文件中加载的URL行列表。
        force_rebuild (bool): 是否强制重新抓取和构建索引，忽略缓存。
        """
        self.url_list = url_lines
        
        # 如果索引文件存在且不强制重建，则直接加载
        if os.path.exists(INDEX_FILE) and not force_rebuild:
            print(f"从 '{INDEX_FILE}' 加载已存在的FAISS索引...")
            self.index = faiss.read_index(str(INDEX_FILE))
            print("索引加载成功。")
            return

        print("正在构建新的FAISS索引...")
        cache = load_cache()
        all_texts = []
        valid_urls = []

        for url_line in self.url_list:
            # 如果强制重建，则不使用缓存
            if force_rebuild or url_line not in cache:
                text = scrape_and_process_url(url_line)
                if text:
                    cache[url_line] = text
            
            # 从缓存中获取文本（可能刚刚存入）
            if url_line in cache:
                all_texts.append(cache[url_line])
                valid_urls.append(url_line)

        # 更新URL列表，只保留成功处理的
        self.url_list = valid_urls
        save_cache(cache)

        if not all_texts:
            print("没有可用于建立索引的有效内容。")
            return
            
        print(f"正在将 {len(all_texts)} 个文档转换为向量...")
        embeddings = self.model.encode(all_texts, show_progress_bar=True)
        
        # 创建FAISS索引
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim) # L2距离索引
        self.index.add(np.array(embeddings, dtype='float32'))
        
        print(f"索引构建完成，共 {self.index.ntotal} 个向量。")
        faiss.write_index(self.index, str(INDEX_FILE))
        print(f"索引已保存到 '{INDEX_FILE}'。")

    def search(self, query, top_k=3):
        """
        根据查询，在FAISS索引中搜索最相关的URL。
        """
        if self.index is None:
            return ["索引尚未构建，请先调用 build_index()。"]
        
        print(f"\n正在执行搜索，查询: '{query}'")
        query_embedding = self.model.encode([query])
        
        # 在FAISS索引中搜索
        distances, indices = self.index.search(np.array(query_embedding, dtype='float32'), top_k)
        
        results = []
        for i in indices[0]:
            if i != -1: # FAISS在结果不足时会返回-1
                # 只返回URL部分
                results.append(self.url_list[i].split()[0])
        
        return results

def load_urls_build_index_search(file_path, query, top_k=3, force_rebuild=False):
    """
    辅助函数：加载URL，构建索引并执行搜索。
    
    参数:
    file_path (str): 包含URL的文件路径。
    query (str): 用户查询。
    top_k (int): 返回的最相关URL数量。
    force_rebuild (bool): 是否强制重新抓取和构建索引。
    """
    retriever = UrlRetriever()
    urls = load_urls_from_file(file_path)
    retriever.build_index(urls, force_rebuild)
    return retriever.search(query, top_k)

# --- 主程序入口 ---
if __name__ == '__main__':
    # 1. 初始化检索器
    retriever = UrlRetriever()
    
    # 2. 加载URL并构建索引
    urls = load_urls_from_file('RAG/urls.txt')
    # 设置 force_rebuild=True 会忽略缓存和旧索引，重新抓取所有网页
    retriever.build_index(urls, force_rebuild=False)
    
    # 3. 执行搜索并展示结果
    # user_query = "where can I learn to code for free?"
    user_query = "what does Obama do?"
    relevant_urls = retriever.search(user_query, top_k=3)
    
    print("\n--- 检索到的最相关URL ---")
    for url in relevant_urls:
        print(url)
        
    # 4. (可选) 为大模型准备Prompt
    # 实际应用中，你可能需要再次抓取这些URL的内容作为上下文
    prompt_for_llm = f"""
    基于以下可能相关的网址，请回答问题。

    相关URL:
    {', '.join(relevant_urls)}

    问题: {user_query}
    """
    print("\n--- 准备好送入大模型的 Prompt (示例) ---")
    print(prompt_for_llm)