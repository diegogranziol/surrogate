from surrogate.tools.search import web_search
from surrogate.tools.fetch import fetch_url


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web. Use for current, local, or time-sensitive "
                "information (restaurants, news, prices, recent events). "
                "Returns a ranked list of title/url/snippet results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return (1-10).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch a URL and return its main readable text (cleaned HTML). "
                "Use after web_search to read the most relevant result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute http(s) URL."}
                },
                "required": ["url"],
            },
        },
    },
]

TOOL_IMPLS = {"web_search": web_search, "fetch_url": fetch_url}
