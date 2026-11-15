"""3-stage LangGraph pipeline: enrich -> retrieve -> predict."""

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from src.pipeline.models import PipelineState
from src.pipeline.nodes import make_enrich_node, make_predict_node, make_retrieve_node


def build_pipeline(
    retrieve_fn: Callable[[str, str, int], list[str]],
    get_records: Callable[[list[str]], dict[str, dict]],
    top_k: int = 5,
):
    """Compile the pipeline with an injected retrieval strategy."""
    graph = StateGraph(PipelineState)
    graph.add_node("enrich", make_enrich_node())
    graph.add_node("retrieve", make_retrieve_node(retrieve_fn, get_records, top_k))
    graph.add_node("predict", make_predict_node())

    graph.add_edge(START, "enrich")
    graph.add_edge("enrich", "retrieve")
    graph.add_edge("retrieve", "predict")
    graph.add_edge("predict", END)

    return graph.compile()
