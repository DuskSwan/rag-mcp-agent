# Usage: uv run main.py
# -*- coding: utf-8 -*-

import asyncio
import argparse

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from src import url_agent_instruction, summarizer_agent_instruction

app = MCPApp(name="web_info_search")

async def main(urls:list[str], query: str):
    async with app.run() as mcp_agent_app:
        logger = mcp_agent_app.logger
        # 创建需要的agent
        fetcher_agent = Agent(
            name="fetcher",
            instruction=url_agent_instruction,
            server_names=["fetch"],  # 声明 agent 可以使用的 mcp server
        )
        summarizer_agent = Agent(
            name="summarizer",
            instruction=summarizer_agent_instruction,
        )
        urls_res = await fetch(urls, fetcher_agent, logger)
        final_result = await summarize(urls_res, query, summarizer_agent, logger)


async def fetch(urls: list[str], agent: Agent, logger):
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
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--url', type=str, required=True, help='The URL to fetch')
    # args = parser.parse_args()
    urls = [
        r'https://en.wikipedia.org/wiki/Barack_Obama',
        r'https://www.britannica.com/biography/Barack-Obama',
        r'https://www.history.com/articles/barack-obama',
    ]
    query = "奥巴马的生平和成就是什么？"
    asyncio.run(main(urls, query))
