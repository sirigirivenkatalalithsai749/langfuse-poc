from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from langfuse import get_client
from tools_exp import web_search
from langfuse import observe
import os

from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


key = os.getenv("GOOGLE_API_KEY")
key2 = os.getenv("GROQ_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_tokens=500,
    api_key=key
)

from langchain_groq import ChatGroq

llm2 = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.7,
    api_key= key2
)


class GraphState(TypedDict):
    query: str
    research: Optional[str]
    summary: Optional[str]
    score: Optional[float]


research_agent = create_agent(
    model=llm,
    tools=[web_search],
    system_prompt=(
        "You are a research agent. Use the web_search tool to gather "
        "accurate and relevant information for the user query."
    ),
)

writer_agent = create_agent(
    model=llm,
    system_prompt=(
        "You are a skilled writer. Summarize the research clearly "
        "and concisely for a general audience."
    ),
)

evaluator_agent = create_agent(
    model=llm2,
    system_prompt=(
        "You are an evaluator. Judge relevance and accuracy of the summary "
        "with respect to the query. Output ONLY a number between 1 and 10."
    ),
)


langfuse = get_client()
callback = CallbackHandler()

@observe(name="research agent",capture_input=True,capture_output=True,as_type="agent")
def research_node(state: GraphState):
    response = research_agent.invoke(
            {"messages": [{"role": "user", "content": state["query"]}]},
            callbacks=[callback],
        )
    research = response["messages"][-1].content
    return {"research": research}

@observe(name="summary agent",capture_input=True,capture_output=True,as_type="generation")
def summary_node(state: GraphState):
    prompt = f"Summarize the following research:\n\n{state['research']}"
    response = writer_agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            callbacks=[callback],
        )
    summary = response["messages"][-1].content
    return {"summary": summary}
@observe(name="evaluator agent", as_type="agent",capture_input=True,capture_output=True)
def evaluator_node(state: GraphState):
    
        prompt = f"""
Query:
{state['query']}

Research:
{state['research']}

Summary:
{state['summary']}

Give a relevance & accuracy score (1–10).
Output ONLY the number.
"""
        response = evaluator_agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            callbacks=[callback],
        )

        score = float(response["messages"][-1].content.strip())

        langfuse.score_current_span(
            name="relevance_score",
            value=score,
            data_type="NUMERIC",
            comment="Relevance and factual accuracy score",
        )

        return {"score": score}


graph = StateGraph(GraphState)

graph.add_node("research", research_node)
graph.add_node("summary", summary_node)
graph.add_node("evaluate", evaluator_node)

graph.add_edge(START, "research")
graph.add_edge("research", "summary")
graph.add_edge("summary", "evaluate")
graph.add_edge("evaluate", END)

compiled_graph = graph.compile()


query = input("Enter your prompt fro research: ")

with langfuse.start_as_current_span(name="POC-Workflow") as root:
    root.update(input={"task":"does the research on user query and then evaluates the responce "})
    final = compiled_graph.invoke({"query": query})
    root.update(output=final)

print("\nSUMMARY:\n", final["summary"])
print("\nSCORE:\n", final["score"])
