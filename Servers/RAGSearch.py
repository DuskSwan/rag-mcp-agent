# -*- coding: utf-8 -*-

import sys

from fastmcp import FastMCP
mcp = FastMCP("rag-search-server")

sys.path.append('.')
from RAG.retriever import load_urls_build_index_search

@mcp.tool()
def search_in_RAG(
    query: str,
    file_path: str ='RAG/urls.txt',
    top_k: int = 3,
    force_rebuild: bool = False
) -> str:
    '''
    search relevant URLs from RAG index.
    Parameters:
        query (str): The search query.
        file_path (str): Path to the file containing URLs.
        top_k (int): Number of top results to return, default is 3.
        force_rebuild (bool): If True, will ignore cache and rebuild index from scratch.
    Returns:
        str: A string containing the top-k relevant URLs, joined by newline characters.
    '''
    try:
        urls =  load_urls_build_index_search(
            file_path=file_path,
            query=query,
            top_k=top_k,
            force_rebuild=force_rebuild
        )
        return '\n'.join(urls)
    except Exception as e:
        return f"Error: {e}"


def test():
    res = load_urls_build_index_search(
        file_path='RAG/urls.txt',
        query="where can I learn to code for free?",
        top_k=3,
        force_rebuild=False
    )
    print(res)
    # zhres = res.encode().decode('unicode_escape')
    # print(zhres)

if __name__ == "__main__":
    # mcp.run(
    #     transport="streamable-http",
    #     host="127.0.0.1",
    #     port=8888,
    #     path="/mcp",
    # )
    mcp.run(transport="stdio")
    # test()
