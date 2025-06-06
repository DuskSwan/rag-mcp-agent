# rag-mcp-agent

我希望搭建一个包含rag和mcp的多智能体框架。该项目用于实现“信息搜索”功能，计划包含三个agent，一个用于从知识库中搜索相关url，一个借助mcp server来从特定url获取信息，最后一个根据获取的信息总结。

## requirements

### 多智能体

使用[mcp-agent](https://github.com/lastmile-ai/mcp-agent)框架

### 网页搜索

自定义mcp server，通过brave search API搜索网页，需要库langchain-community，fastmcp

从网页获取内容的server fetch来自第三方，使用指令`uvx mcp-server-fetch`

### 知识库

用sentence_transformers库来实现知识库的检索，这部分代码是Gemini写的
