# mcp_agent_demo.py
# ---------------------------------------------------------------
# Connects a CrewAI agent to our MCP server (mcp_math_server.py)
# and asks it to use the MCP "add" tool.
#
# Run it with:   python mcp_agent_demo.py
# ---------------------------------------------------------------

import sys
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

# How to start our MCP server (run the server file with Python)
server_params = StdioServerParameters(command=sys.executable, args=["mcp_math_server.py"])

# Connect to the server. 'mcp_tools' are the tools it shares.
with MCPServerAdapter(server_params) as mcp_tools:
    print("Tools the agent got from MCP:", [t.name for t in mcp_tools])

    agent = Agent(
        role="Math Helper",
        goal="Solve math using the MCP tools",
        backstory="You always use the tools from the MCP server.",
        tools=mcp_tools,                 # <-- MCP tools given to the agent
        llm=LLM(model="gpt-4o-mini"),
    )

    task = Task(
        description="What is 8 plus 5? Use your tool to add them.",
        expected_output="The number answer.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()              # plain .py script -> kickoff() is fine here
    print("Answer:", result)
