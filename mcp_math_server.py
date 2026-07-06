# mcp_math_server.py
# ---------------------------------------------------------------
# A tiny MCP server. It offers ONE tool to any AI agent: "add".
# An MCP server is just a small program that shares tools in a
# standard way, so any agent can plug in and use them.
# ---------------------------------------------------------------

from mcp.server.fastmcp import FastMCP

# Give the server a name
mcp = FastMCP("MathServer")


# Any function with @mcp.tool() becomes a tool the agent can use.
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b


# Start the server (talks to the agent through stdio = standard input/output)
if __name__ == "__main__":
    mcp.run(transport="stdio")
