import os
from typing import TypedDict, Literal
from dotenv import load_dotenv   # 修正导入
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

load_dotenv()

class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str

llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.7,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)

class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="判断笑话是否幽默",
        examples=["funny", "not funny"]
    )
    feedback: str = Field(
        description="若不幽默，提供改进建议",
        examples=["可以加入双关语或意外结局"]
    )

def generator_func(state: State) -> dict:
    if state.get("feedback"):
        prompt = f"请根据以下反馈改进关于{state['topic']}的笑话：{state['feedback']}\n输出改进后的笑话："
    else:
        prompt = f"请创作一个关于{state['topic']}的冷笑话。"
    chain = llm | StrOutputParser()
    resp = chain.invoke(prompt)
    return {"joke": resp}

def evaluator_func(state: State) -> dict:
    chain = llm.with_structured_output(Feedback)
    resp = chain.invoke(f"评估此笑话的幽默程度：\n{state['joke']}\n注意：幽默应包含意外性或巧妙措辞")
    return {
        "funny_or_not": resp.grade,
        "feedback": resp.feedback
    }

def route_func(state: State) -> Literal["generator", "__end__"]:
    return END if state.get("funny_or_not") == "funny" else "generator"

builder = StateGraph(State)
builder.add_node("generator", generator_func)
builder.add_node("evaluator", evaluator_func)

builder.add_edge(START, "generator")
builder.add_edge("generator", "evaluator")
builder.add_conditional_edges("evaluator", route_func, {
    "generator": "generator",
    "__end__": END
})

graph = builder.compile()

# 测试运行
if __name__ == "__main__":
    initial_state = {
        "joke": "",
        "topic": "黑人",
        "feedback": "",
        "funny_or_not": ""
    }
    final = graph.invoke(initial_state)
    print("最终笑话:", final["joke"])