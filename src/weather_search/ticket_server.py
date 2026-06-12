import asyncio
import sys
from fastmcp import FastMCP

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

mcp = FastMCP("Ticket Query")

@mcp.tool()
async def search_tickets(from_station: str, to_station: str, date: str) -> str:
    """查询火车票（当前为模拟数据，可接入真实 12306 API）"""
    # 模拟数据，实际可接入 12306 开放 API
    return f"从 {from_station} 到 {to_station} 于 {date} 的列车有余票（模拟数据）。如需真实数据，请后续接入 API。"

if __name__ == "__main__":
    mcp.run(transport="stdio")
