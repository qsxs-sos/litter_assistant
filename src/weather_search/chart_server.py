import asyncio
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from fastmcp import FastMCP

mcp = FastMCP("Chart Generator")

@mcp.tool()
def generate_line_chart(data_csv: str, x_label: str = "X", y_label: str = "Y", title: str = "Chart") -> str:
    try:
        data = pd.read_csv(pd.compat.StringIO(data_csv))
        if len(data.columns) < 2:
            return "Error: CSV must have at least two columns."
        x_col, y_col = data.columns[0], data.columns[1]
        plt.figure(figsize=(10,6))
        plt.plot(data[x_col], data[y_col], marker='o')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        os.makedirs("charts", exist_ok=True)
        save_path = "charts/generated_chart.png"
        plt.savefig(save_path)
        plt.close()
        return f"Chart saved to {save_path}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")