import os
import sys
import asyncio
import re
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END, START, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.postgres import AsyncPostgresStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.7,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)

DB_URI = os.getenv("DATABASE_URI", "postgresql://postgres:root@localhost:5432/langgraph_db")

class AssistantState(TypedDict):
    messages: Annotated[list, add_messages]

async def get_long_term_memory(store: AsyncPostgresStore, user_id: str):
    namespace = ("user_memory", user_id)
    item = await store.aget(namespace, "data")
    return item.value if item else {}

async def update_long_term_memory(store: AsyncPostgresStore, user_id: str, updates: dict):
    namespace = ("user_memory", user_id)
    current = await get_long_term_memory(store, user_id)
    current.update(updates)
    await store.aput(namespace, "data", current)

async def agent_node(state: AssistantState, config: RunnableConfig):
    store = config.get("configurable", {}).get("store")
    if not store:
        raise ValueError("Store not found in config['configurable']")
    user_id = config["configurable"]["thread_id"]
    long_mem = await get_long_term_memory(store, user_id)

    last_msg = state["messages"][-1].content
    updates = {}

    if "我叫" in last_msg:
        name_match = re.search(r"我叫([\u4e00-\u9fa5]{2,4})", last_msg)
        if name_match:
            updates["name"] = name_match.group(1)
            print(f"📝 更新长期记忆: name={updates['name']}")
    if "我喜欢" in last_msg:
        hobby_match = re.search(r"我喜欢([\u4e00-\u9fa5\s]{2,10})", last_msg)
        if hobby_match:
            updates["hobby"] = hobby_match.group(1)
            print(f"📝 更新长期记忆: hobby={updates['hobby']}")

    if updates:
        await update_long_term_memory(store, user_id, updates)
        long_mem.update(updates)

    sys_prompt = "你是智能助手，可以调用工具进行数据获取、图表生成和火车票查询。"
    if long_mem.get("name"):
        sys_prompt += f" 用户叫 {long_mem['name']}，请友好称呼。"
    if long_mem.get("hobby"):
        sys_prompt += f" 用户喜欢 {long_mem['hobby']}，可适当提及。"

    messages = [SystemMessage(content=sys_prompt)] + state["messages"]
    tools = config.get("tools", [])
    llm_with_tools = llm.bind_tools(tools)
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

async def tool_node(state: AssistantState, config: RunnableConfig):
    last_msg = state["messages"][-1]
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {}
    results = []
    tools_map = config.get("tools_map", {})
    for tc in last_msg.tool_calls:
        tool_name = tc["name"]
        args = tc["args"]
        tool_func = tools_map.get(tool_name)
        if tool_func:
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**args)
            else:
                result = tool_func(**args)
            results.append(ToolMessage(content=str(result), tool_call_id=tc["id"], name=tool_name))
        else:
            results.append(ToolMessage(content=f"Tool {tool_name} not found", tool_call_id=tc["id"], name=tool_name))
    return {"messages": results}

async def human_intervention(state: AssistantState, config: RunnableConfig):
    print("\n⚠️ 需要人工确认：是否继续执行？")
    decision = await asyncio.to_thread(input, "输入 y 继续，n 取消: ")
    if decision.lower() == 'n':
        return {"messages": [AIMessage(content="任务已取消")]}
    return {"messages": [AIMessage(content="继续执行")]}

INTERVENTION_TOOLS = ["fetch_web_data", "generate_line_chart", "search_tickets"]

def should_continue(state: AssistantState) -> Literal["tools", "human_intervention", "__end__"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        tool_names = [tc["name"] for tc in last_msg.tool_calls]
        if any(name in INTERVENTION_TOOLS for name in tool_names):
            return "human_intervention"
        return "tools"
    return "__end__"

def route_after_intervention(state: AssistantState) -> Literal["agent", "__end__"]:
    last_msg = state["messages"][-1]
    if last_msg.content == "继续执行":
        return "agent"
    return "__end__"

async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    mcp_servers = {
        "chart": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [os.path.join(current_dir, "chart_server.py")],
        },
        "web": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [os.path.join(current_dir, "web_server.py")],
        },
        "ticket": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [os.path.join(current_dir, "ticket_server.py")],
        },
    }

    try:
        client = MultiServerMCPClient(mcp_servers)
        tools = await client.get_tools()
        tools_map = {tool.name: tool for tool in tools}
        print(f"✅ 加载了 {len(tools)} 个 MCP 工具: {list(tools_map.keys())}")
    except Exception as e:
        print(f"❌ 加载 MCP 工具失败: {e}")
        return

    # 将 PostgreSQL 连接的生命周期包含整个交互过程
    async with AsyncPostgresStore.from_conn_string(DB_URI) as store:
        await store.setup()
        print("✅ PostgreSQL 长期存储已连接")

        checkpointer = MemorySaver()
        builder = StateGraph(AssistantState)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", tool_node)
        builder.add_node("human_intervention", human_intervention)

        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", should_continue, {
            "tools": "tools",
            "human_intervention": "human_intervention",
            "__end__": END
        })
        builder.add_conditional_edges("human_intervention", route_after_intervention, {
            "agent": "agent",
            "__end__": END
        })
        builder.add_edge("tools", "agent")

        graph = builder.compile(checkpointer=checkpointer, store=store)

        thread_id = input("请输入会话ID（例如 user_001）: ").strip() or "user_001"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "store": store   # 传递 store 引用
            },
            "tools_map": tools_map,
            "tools": tools,
        }

        print("\n🤖 智能小秘书已启动（输入 quit 退出）")
        while True:
            user_input = input("\n用户: ")
            if user_input.lower() == "quit":
                break
            try:
                result = await graph.ainvoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config
                )
                final_msg = result["messages"][-1].content if result["messages"] else "无回复"
                print(f"助手: {final_msg}")
            except Exception as e:
                print(f"执行出错: {e}")

if __name__ == "__main__":
    asyncio.run(main())
