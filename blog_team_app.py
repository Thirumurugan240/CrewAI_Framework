# blog_team_app.py
# ---------------------------------------------------------------
# A simple Blog Team built with CrewAI + Streamlit.
#
# 3 agents work together:
#   1. Trend Researcher -> finds a trending subtopic for your topic
#   2. Content Writer   -> writes the blog post
#   3. SEO Specialist   -> improves the post for search engines
# ---------------------------------------------------------------
#
# SETUP (run once in the terminal):
#   pip install crewai crewai-tools streamlit python-dotenv
#
#   Put your keys in a .env file next to this script:
#     OPENAI_API_KEY=your-openai-key
#     SERPER_API_KEY=your-serper-key      (free from https://serper.dev - used for web search)
#
# RUN:
#   streamlit run blog_team_app.py
# ---------------------------------------------------------------

import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

# Load the keys from the .env file
load_dotenv()

# The AI brain used by all three agents
llm = LLM(model="gpt-4o-mini")

# The web search tool (lets the researcher find current trends online)
search_tool = SerperDevTool()


# ---------- The webpage ----------
st.title("📝 Blog Team (CrewAI)")
st.write("Enter a topic. Three AI agents will research, write, and SEO-optimize a blog for you.")

topic = st.text_input("Enter your blog topic:", "Healthy eating")

if st.button("Generate Blog"):

    # ----- Agent 1: find a trending subtopic -----
    researcher = Agent(
        role="Trend Researcher",
        goal="Search the web and find one trending subtopic for the given topic",
        backstory="You search the internet to see what is popular right now.",
        tools=[search_tool],     # <-- can now search the web
        llm=llm,
    )

    # ----- Agent 2: write the blog -----
    writer = Agent(
        role="Content Writer",
        goal="Write a simple, friendly blog post",
        backstory="You write clear blogs that beginners enjoy reading.",
        llm=llm,
    )

    # ----- Agent 3: do SEO on the blog -----
    seo = Agent(
        role="SEO Specialist",
        goal="Make the blog easy to find on search engines",
        backstory="You add catchy titles, keywords, and meta descriptions.",
        llm=llm,
    )

    # ----- Their tasks (run in order) -----
    research_task = Task(
        description=f"Search the web and find ONE trending subtopic for the topic: {topic}.",
        expected_output="One trending subtopic and a short reason why it is popular.",
        agent=researcher,
    )

    write_task = Task(
        description="Write a short, simple blog post (about 200 words) about that trending subtopic.",
        expected_output="A blog post with a title and a few short paragraphs.",
        agent=writer,
    )

    seo_task = Task(
        description="Improve the blog for SEO. Add a catchy title, 5 keywords, and a 1-line meta description.",
        expected_output="The final blog with a title, keywords, meta description, and the post.",
        agent=seo,
    )

    # ----- Put the team together and run -----
    crew = Crew(
        agents=[researcher, writer, seo],
        tasks=[research_task, write_task, seo_task],
    )

    with st.spinner("The blog team is working... please wait."):
        result = crew.kickoff()

    # ----- Show the final blog -----
    st.subheader("Your SEO-Optimized Blog")
    st.markdown(result.raw)
