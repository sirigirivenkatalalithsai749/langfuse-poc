from ddgs import DDGS
from langfuse import get_client

langfuse = get_client()

def web_search(query: str) -> str:
    """
    DuckDuckGo web search tool with LangFuse span tracing.
    """

    with langfuse.start_as_current_span(name="tool:web_search") as span:
        span.update(input={"query": query})

        results = []

        with DDGS() as ddgs:
            for r in ddgs.text(
                query,
                max_results=5,
                safesearch="moderate",
                region="us-en",
            ):
                title = r.get("title", "")
                body = r.get("body", "")
                results.append(f"- {title}: {body}")

        if not results:
            output = "No results found from DuckDuckGo."
        else:
            output = "\n".join(results)

        span.update(output=output)

        return output
