# Usage: uv run main.py
# -*- coding: utf-8 -*-

import asyncio
import argparse

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from prompts import url_agent_instruction, summarizer_agent_instruction, search_agent_instruction

app = MCPApp(name="web_info_search")

async def main(query: str):
    async with app.run() as mcp_agent_app:
        logger = mcp_agent_app.logger
        # 创建需要的agent
        searcher_agent = Agent(
            name="searcher",
            instruction=search_agent_instruction,
            server_names=["websearch"],  # 声明 agent 可以使用的 mcp server
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
        search_result = await search(query, searcher_agent, logger)
        urls_res = await fetch(search_result, fetcher_agent, logger)
        final_result = await summarize(urls_res, query, summarizer_agent, logger)

        logger.info(f"最终结果: \n{final_result}")

async def search(query: str, agent: Agent, logger):
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
        logger.info(f"搜索结果: \n{result}")
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
        logger.info(f"url读取结果: \n{result}")
    return result

async def summarize(content: str, query: str, agent: Agent, logger):
    async with agent:
        llm = await agent.attach_llm(OpenAIAugmentedLLM)
        result = await llm.generate_str(
            message=f"请根据用户的问题【{query}】，从以下内容中总结出合适的回答: {content}"
        )
        logger.info(f"总结结果: \n{result}")
    return result

if __name__ == "__main__":
    query = "What is Obama's life and achievements?"
    asyncio.run(main(query))
