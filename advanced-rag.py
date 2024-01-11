import os
from typing import List

from dotenv import load_dotenv
from llama_index import (Document, ServiceContext, SimpleDirectoryReader,
                         StorageContext, VectorStoreIndex)
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings import GeminiEmbedding, OpenAIEmbedding
from llama_index.indices.query.query_transform.base import \
    StepDecomposeQueryTransform
from llama_index.llms import Gemini, OpenAI
from llama_index.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.query_engine import (MultiStepQueryEngine,
                                      RetrieverQueryEngine,
                                      SubQuestionQueryEngine,
                                      retriever_query_engine)
from llama_index.readers.database import DatabaseReader
from llama_index.retrievers.auto_merging_retriever import AutoMergingRetriever
from llama_index.schema import BaseNode
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.tools.types import ToolMetadata

load_dotenv()


def load_documents() -> List[Document]:
    # Using Database
    loader = DatabaseReader(uri=connection_string)
    documents = loader.load_data(query=query)  # INFO: Ignore this

    # Using directory
    # loader = SimpleDirectoryReader(input_files=["./documents/_event__202401101443.csv"])
    # documents = loader.load_data(show_progress=True)

    documents = Document(text="\n\n".join([doc.get_content() for doc in documents]))
    return [documents]


def build_nodes(documents: List[Document]) -> List[BaseNode]:
    """
    Take a list of documents and chunk them into `Node` objects, such that
    each node is a specific chunk of parent document.
    """

    # node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    # node_parser = TokenTextSplitter(chunk_size=512, chunk_overlap=10, separator="\n\n")
    # node_parser = JSONNodeParser()
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[1024, 512, 256, 128]
    )

    nodes = node_parser.get_nodes_from_documents(
        documents=documents, show_progress=True
    )
    return nodes


def build_storage_context(nodes: List[BaseNode]) -> StorageContext:
    # contain ingested document chunks (Node)
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    docstore.persist("./storage/docstore.json")
    # docstore = SimpleDocumentStore.from_persist_dir("./storage")

    storage_context = StorageContext.from_defaults(docstore=docstore)
    return storage_context


def build_service_context() -> ServiceContext:
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        callback_manager=callback_manager,
    )
    return service_context


def build_index(
    nodes: List[BaseNode],
    service_context: ServiceContext,
    storage_context: StorageContext,
) -> VectorStoreIndex:
    # index = (
    #     VectorStoreIndex.from_vector_store(vector_store=vector_store)
    #     if is_exist
    #     else VectorStoreIndex(
    #         nodes=leaf_nodes,
    #         service_context=service_context,
    #         storage_context=storage_context,
    #         show_progress=True,
    #     )
    # )
    index = VectorStoreIndex(
        nodes=nodes,
        service_context=service_context,
        storage_context=storage_context,
        show_progress=True,
    )
    return index


def build_retriever(
    index: VectorStoreIndex,
    storage_context: StorageContext,
):
    base_retriever = index.as_retriever(similarity_top_k=6)
    retriever = AutoMergingRetriever(
        vector_retriever=base_retriever,  # INFO: Ignore this
        storage_context=storage_context,
        verbose=True,
        callback_manager=callback_manager,
    )
    return retriever


def main() -> None:
    documents = load_documents()
    nodes = build_nodes(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    storage_context = build_storage_context(nodes)
    service_context = build_service_context()

    index = build_index(
        nodes=leaf_nodes,
        service_context=service_context,
        storage_context=storage_context,
    )

    retriever = build_retriever(index=index, storage_context=storage_context)
    query_engine = RetrieverQueryEngine.from_args(retriever=retriever)

    # query_engine = MultiStepQueryEngine(
    #     query_engine=query_engine,
    #     query_transform=StepDecomposeQueryTransform(llm=llm, verbose=True),
    # )
    response = query_engine.query(query_str)
    print(response)


if __name__ == "__main__":
    connection_string = os.getenv("CONNECTION_STRING")
    query = os.getenv("QUERY")

    llm = Gemini(model_name="models/gemini-pro", temperature=0.1)
    # llm = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)
    # embed_model = GeminiEmbedding()
    embed_model = OpenAIEmbedding()

    callback_manager = CallbackManager(
        [
            LlamaDebugHandler(print_trace_on_end=True),
        ]
    )

    query_str = "Give me all events in Hawaii"
    # query_str = "Give me all events which held in Hawaii. Return event id array only."

    main()
