from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from langfuse import get_client
from tools import web_search
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


def research_node(state: GraphState):
    with langfuse.start_as_current_span(name="research_agent") as span:
        response = research_agent.invoke(
            {"messages": [{"role": "user", "content": state["query"]}]},
            callbacks=[callback],
        )
        research = response["messages"][-1].content
        span.update(output=research)

    return {"research": research}

def summary_node(state: GraphState):
    with langfuse.start_as_current_generation(name="summary_agent") as gen:
        prompt = f"Summarize the following research:\n\n{state['research']}"
        response = writer_agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            callbacks=[callback],
        )
        summary = response["messages"][-1].content
        gen.update(output=summary)

    return {"summary": summary}

def evaluator_node(state: GraphState):
    with langfuse.start_as_current_span(name="evaluator_agent") as span:
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

        span.update(output=score)

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


query = input("Enter your question: ")

with langfuse.start_as_current_span(name={"task":"do research on the query and evaluate it"}) as root:
    final = compiled_graph.invoke({"query": query})
    root.update(output=final)

print("\nSUMMARY:\n", final["summary"])
print("\nSCORE:\n", final["score"])
