__all__ = ["get_query_formatter"]


def get_query_formatter(query_type):
    if query_type == "completion_agent":
        return lambda query: query
    elif query_type == "instructional_agent":
        return lambda query: [{"role": "user", "content": query}]
    else:
        return "llama3"
    
