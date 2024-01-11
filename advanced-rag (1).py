import logging
import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from dotenv import load_dotenv
from llama_index import (OpenAIEmbedding, ServiceContext, StorageContext,
                         VectorStoreIndex)
from llama_index.indices.query.query_transform import HyDEQueryTransform
from llama_index.llms import OpenAI
from llama_index.query_engine.retriever_query_engine import \
    RetrieverQueryEngine
from llama_index.query_engine.transform_query_engine import \
    TransformQueryEngine
from llama_index.retrievers.auto_merging_retriever import AutoMergingRetriever
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores import PGVectorStore

from advanced_rag.document_reader import DocumentReader
from advanced_rag.node_parser import NodeParser


def main():
    # Step 1: Read document
    document = DocumentReader(
        connection_string=connection_string, query=query
    ).get_document()
    index = VectorStoreIndex.from_documents(documents=[document])
    query_engine = index.as_query_engine()

    # Step 2: Parse document into nodes
    # (nodes, leaf_nodes) = NodeParser(
    #     document=document, chunk_sizes=[512, 256, 128], show_progress=True
    # ).get_nodes()
    # docstore.add_documents(nodes=nodes)

    # for node in nodes:
    #     print(node.get_content())
    #     print("\n" + "#" * 50 + "\n")

    # Step 3: Create Index
    # service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    # storage_context = StorageContext.from_defaults(
    #     docstore=docstore,
    #     # vector_store=vector_store,
    # )
    # index = VectorStoreIndex(
    #     nodes=leaf_nodes,
    #     storage_context=storage_context,
    #     service_context=service_context,
    #     show_progress=True,
    # )
    # vector_retriever = index.as_retriever(similarity_top_k=10)
    #
    # # Step 4: Create query engine
    # retriever = AutoMergingRetriever(
    #     vector_retriever=vector_retriever,  # INFO: Ignore this
    #     storage_context=storage_context,
    #     verbose=True,
    # )
    # hyde = HyDEQueryTransform(include_original=True)
    # query_engine = RetrieverQueryEngine.from_args(retriever=retriever, streaming=True)
    # query_engine = TransformQueryEngine(query_engine=query_engine, query_transform=hyde)
    #
    # # Step 5: Feed to LLM
    # retrieval_nodes = retriever.retrieve(query_str)
    # for node in retrieval_nodes:
    #     print(
    #         "Node ID:",
    #         node.id_,
    #         "\nSimilarity:",
    #         node.score,
    #         "\nText:",
    #         node.get_content(),
    #     )
    #     print("\n" + "#" * 50 + "\n")
    # print()
    # response = query_engine.query(query_str)
    # response.print_response_stream()  # INFO: Ignore this


if __name__ == "__main__":
    load_dotenv()
    connection_string = os.getenv("CONNECTION_STRING")
    query = os.getenv("QUERY")

    llm = OpenAI(
        # model="gpt-3.5-turbo-1106",
        model="gpt-4",
        max_tokens=2048,
        temperature=0.2,
    )
    embed_model = OpenAIEmbedding()
    docstore = SimpleDocumentStore()
    # vector_store = PGVectorStore.from_params(
    #     connection_string=connection_string,
    #     port="5433",  # WARN: Must set this constant
    # )
    # query_str = "Give me total number of events that had vanue in Hawaii"
    query_str = "Give me all events that took place in exotic destination."

    main()
