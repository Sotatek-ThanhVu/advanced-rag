import os
from typing import Any, List, Tuple

from dotenv import load_dotenv
from llama_index import (OpenAIEmbedding, ServiceContext, StorageContext,
                         load_index_from_storage)
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings import GeminiEmbedding
from llama_index.llms import Gemini, OpenAI
from llama_index.llms.utils import LLMType
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import ResponseMode
from llama_index.retrievers.auto_merging_retriever import AutoMergingRetriever
from llama_index.schema import NodeWithScore

load_dotenv()
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def display_nodes(nodes: List[NodeWithScore]):
    for node in nodes:
        print("\n" + "-" * 50 + "\n")
        print("Node ID:", node.id_)
        print("Node score:", node.score)
        print("Node content:", node.text)
        print("\n" + "-" * 50 + "\n")


def build_context(
    llm: LLMType | None,
    embed_model: Any | None,
    callback_manager: CallbackManager | None,
    persist_dir: str = "storage",
) -> Tuple[ServiceContext, StorageContext]:
    """
    Construct ServiceContext and StorageContext
    """

    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, callback_manager=callback_manager
    )

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

    return service_context, storage_context


def main() -> None:
    service_context, storage_context = build_context(llm, embed_model, callback_manager)

    index = load_index_from_storage(storage_context, service_context=service_context)
    base_retriever = index.as_retriever(similarity_top_k=10)

    retriever = AutoMergingRetriever(
        base_retriever,
        storage_context=storage_context,
        verbose=True,
        callback_manager=callback_manager,
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        # response_mode=ResponseMode.NO_TEXT,
        streaming=True,
    )

    response = query_engine.query(query)
    # display_nodes(response.source_nodes)
    response.print_response_stream()


if __name__ == "__main__":
    connection_string = os.getenv("CONNECTION_STRING")
    database_query = os.getenv("QUERY")

    # llm = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.0, max_tokens=250)
    # embed_model = OpenAIEmbedding(mode=OpenAIEmbeddingMode.SIMILARITY_MODE)
    llm = Gemini(temperature=0.0, max_tokens=512)
    embed_model = GeminiEmbedding()
    callback_manager = CallbackManager([LlamaDebugHandler(print_trace_on_end=True)])

    # query = "Give me all events id taking place in Hawaii. No description."
    query = "What events are held in Hawaii? Return in array of id only."

    main()
