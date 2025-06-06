# Usage: uv run main.py
# -*- coding: utf-8 -*-

import asyncio

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM

from prompts import url_agent_instruction, summarizer_agent_instruction, web_search_agent_instruction, rag_search_agent_instruction

app = MCPApp(name="web_info_search")

async def main(query: str):
    # TODO: 将初始化RAG挪到main中
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
        parallel = ParallelLLM(
            fan_in_agent=fetcher_agent,
            fan_out_agents=[rag_searcher_agent, web_searcher_agent],
            llm_factory=OpenAIAugmentedLLM,
        )
        # 工作流
        context_res = await parallel.generate_str(
            message=f"根据用户查询【{query}】寻找合适的url，然后获取信息。",
        )
        logger.info(f"获取到的相关内容: \n{context_res}")
        summarizer_llm = await summarizer_agent.attach_llm(OpenAIAugmentedLLM)
        final_result = await summarizer_llm.generate_str(
            message=f"请根据用户的问题【{query}】，从以下内容中总结出合适的回答: {context_res}"
        )
        logger.info(f"最终结果: \n{final_result}")

if __name__ == "__main__":
    query = "What is Obama's life and achievements?"
    asyncio.run(main(query))
