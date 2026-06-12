import os
from typing import TypedDict, Literal

import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

load_dotenv()


class State(TypedDict):
    joke: str  # 生成冷笑话内容
    topic: str  # 主题
    feedback: str  # 改进建议
    funny_or_not: str  # 幽默评级


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


# 定义节点函数(笑话节点)
def generator_func(state: State) -> State:
    """由大模型生成一个冷笑话节点"""
    prompt = (
        f"根据反馈改进笑话{state['feedback']}\n主题: {state['topic']}"
        if state.get("feedback")
        else f"创作一个关于{state['topic']}的笑话"
    )
    # 第一种写法
    # response = llm.invoke(prompt)
    # return {'joke': response.content}
    # 第二种 （使用输出解压器）:会把内容StrOutputParser() 按照解析器进行解析。
    chain = llm | StrOutputParser()
    resp = chain.invoke(prompt)
    return {'joke': resp}


# 定义节点函数（评估节点）
def avaluator_func(state: State) -> State:
    """评估状态中的冷笑话"""
    prompt = (
        f"根据反馈改进笑话{state['feedback']}\n主题: {state['topic']}"
        if state.get("feedback")
        else f"创作一个关于{state['topic']}的笑话"
    )

    # 第二种 （使用输出解压器）:会把内容StrOutputParser() 按照解析器进行解析。
    chain = llm | StrOutputParser()
    resp = chain.invoke(prompt)
    return {'joke': resp}


def avaluator_func1(state: State):
    # """评估状态中的冷笑话"""
    chain = llm.with_structured_output(Feedback)
    resp = chain.invoke(
        f"评估此笑话的幽默程度：\n{state['joke']}\n"
        "注意：幽默应包换意外性或巧妙措辞"
    )
    return {'grade': resp.grade,
            'feedback': resp.feedback}


# 第二种，仿制成工具模型
def avaluator_func2(state: State):
    # """评估状态中的冷笑话"""
    chain = llm.with_structured_output(Feedback)
    evaluation = chain.invoke(
        f"评估此笑话的幽默程度：\n{state['joke']}\n"
        "注意：幽默应包换意外性或巧妙措辞"
    )
    evaluation = evaluation.tool_calls[-1]['args']
    return {
        "funny_or_not": evaluation['grade'],
        "feedback": evaluation['feedback']
    }



# 构建一个工作流
builder = StateGraph(State)

builder.add_node('generator', generator_func)
builder.add_node('avaluator', avaluator_func1)

#
