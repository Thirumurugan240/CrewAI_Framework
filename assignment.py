import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

llm = LLM(model="gpt-4o-mini")

# 1. Build RAG
def build_vector_db():
    loader = DirectoryLoader(
        "data",
        glob="**/*.txt",
        loader_cls=TextLoader
    )

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db


vector_db = build_vector_db()


# 2. Create RAG Tool

@tool("Customer Support Knowledge Base")
def customer_support_rag_tool(query: str) -> str:
    """
    Use this tool to search company FAQs, policies, pricing, refund rules,
    product documentation, and customer support knowledge base.
    """
    results = vector_db.similarity_search(query, k=4)

    context = "\n\n".join([
        doc.page_content for doc in results
    ])

    return context


# 3. Create Customer Support Agent
support_agent = Agent(
    role="Customer Support Agent",
    goal="Answer customer queries accurately using the knowledge base.",
    backstory=(
        "You are a helpful customer support agent. "
        "You must answer only using the provided knowledge base. "
        "If the answer is not available, politely say that you don't have enough information "
        "and suggest contacting the support team."
    ),
    tools=[customer_support_rag_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)


# 4. Run Query
def answer_customer(query: str):
    task = Task(
        description=f"""
        Customer asked:

        {query}

        Search the knowledge base using the RAG tool.
        Give a clear, friendly, and professional customer support answer.
        Do not hallucinate.
        """,
        expected_output="A helpful customer support response based only on the knowledge base.",
        agent=support_agent
    )

    crew = Crew(
        agents=[support_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    return result


if __name__ == "__main__":
    query = input("Customer Query: ")
    response = answer_customer(query)
    print("\nAgent Response:\n")
    print(response)