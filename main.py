# Usage: uv run main.py
# -*- coding: utf-8 -*-

import asyncio

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from prompts import url_agent_instruction, summarizer_agent_instruction, web_search_agent_instruction, rag_search_agent_instruction

app = MCPApp(name="web_info_search")

async def main(query: str):
    # 初始化RAG

    # 调用agent
    async with app.run() as mcp_agent_app:
        logger = mcp_agent_app.logger
        # 创建需要的agent
        web_searcher_agent = Agent(
            name="web_searcher",
            instruction=web_search_agent_instruction,
            server_names=["webSearch"],  # 声明 agent 可以使用的 mcp server
        )
        rag_searcher_agent = Agent(
            name="rag_searcher",
            instruction=rag_search_agent_instruction,
            server_names=["ragSearch"],  # 声明 agent 可以使用的 mcp server
        )
        fetcher_agent = Agent(
            name="fetcher",
            instruction=url_agent_instruction,
            server_names=["fetch"],  # 声明 agent 可以使用的 mcp server
        )
        summarizer_agent = Agent(
            name="summarizer",
            instruction=summarizer_agent_instruction,
        )
        # 执行过程
        web_search_result = await web_search(query, web_searcher_agent, logger)
        rag_search_result = await rag_search(query, rag_searcher_agent, logger)
        merged_result = web_search_result + rag_search_result
        urls_res = await fetch(merged_result, fetcher_agent, logger)
        final_result = await summarize(urls_res, query, summarizer_agent, logger)

        logger.info(f"最终结果: \n{final_result}")

async def web_search(query: str, agent: Agent, logger):
    async with agent:
        # 确保 MCP Server 初始化完成, 可以被 LLM 使用
        tools = await agent.list_tools()
        logger.info("可用工具:", data=tools)

        # Attach an OpenAI LLM to the agent
        llm = await agent.attach_llm(OpenAIAugmentedLLM)

        # 使用 MCP Server -> websearch 获取相关网页链接
        result = await llm.generate_str(
            message=f"请根据用户的查询【{query}】返回相关的网页链接列表。"
        )
        # logger.info(f"搜索结果: \n{result}")
    return result

async def rag_search(query: str, agent: Agent, logger):
    async with agent:
        # 确保 MCP Server 初始化完成, 可以被 LLM 使用
        tools = await agent.list_tools()
        logger.info("可用工具:", data=tools)

        # Attach an OpenAI LLM to the agent
        llm = await agent.attach_llm(OpenAIAugmentedLLM)

        # 使用 MCP Server -> RAG Search 获取相关网页链接
        result = await llm.generate_str(
            message=f"请根据用户的查询【{query}】返回相关的网页链接列表。"
        )
        # logger.info(f"RAG搜索结果: \n{result}")
    return result

async def fetch(urls: str, agent: Agent, logger):
    async with agent:
        # 确保 MCP Server 初始化完成, 可以被 LLM 使用
        tools = await agent.list_tools()
        logger.info("可用工具:", data=tools)

        # Attach an OpenAI LLM to the agent
        llm = await agent.attach_llm(OpenAIAugmentedLLM)

        # 使用 MCP Server -> fetch 获取指定 URL 网页内容
        result = await llm.generate_str(
            message=f"从以下url获取信息: {urls}"
        )
        # logger.info(f"url读取结果: \n{result}")
    return result

async def summarize(content: str, query: str, agent: Agent, logger):
    async with agent:
        llm = await agent.attach_llm(OpenAIAugmentedLLM)
        result = await llm.generate_str(
            message=f"请根据用户的问题【{query}】，从以下内容中总结出合适的回答: {content}"
        )
        # logger.info(f"总结结果: \n{result}")
    return result

if __name__ == "__main__":
    query = "What is Obama's life and achievements?"
    asyncio.run(main(query))
