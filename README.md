# rag-mcp-agent

一个包含rag和mcp的多智能体框架。该项目用于实现“信息搜索”功能，总共包含四个agent，A用于从知识库中搜索相关url，B通过联网搜索获取相关url，C从特定url获取信息，D根据获取的信息总结出最终结果。AB并行，\[AB\]CD串行。

## requirements

### 多智能体

使用[mcp-agent](https://github.com/lastmile-ai/mcp-agent)框架

### 网页搜索

自定义mcp server，通过brave search API搜索网页，需要库langchain-community，fastmcp

从网页获取内容的server fetch来自第三方，使用指令`uvx mcp-server-fetch`

### 知识库

用sentence_transformers库来实现知识库的检索，这部分代码是Gemini写的。需要库faiss-cpu，sentence-transformers
