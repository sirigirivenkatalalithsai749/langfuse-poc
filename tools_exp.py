from ddgs import DDGS
from langfuse import get_client , observe

langfuse = get_client()
@observe(name="web search tool", as_type="tool",capture_input=True,capture_output=True)
def web_search(query: str) -> str:
    """
    DuckDuckGo web search tool with LangFuse span tracing.
    """

    

    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(
            query,
            max_results=5,
            safesearch="moderate",
            region="us-en"):
                
                title = r.get("title", "")
                body = r.get("body", "")
                results.append(f"- {title}: {body}")

        if not results:
            output = "No results found from DuckDuckGo."
        else:
            output = "\n".join(results)

      

        return output
