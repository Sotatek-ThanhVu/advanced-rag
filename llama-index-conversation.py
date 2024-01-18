import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from llama_index import (PromptTemplate, QueryBundle, ServiceContext,
                         VectorStoreIndex, set_global_service_context)
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.chat_engine import CondensePlusContextChatEngine
from llama_index.embeddings import GeminiEmbedding
from llama_index.llms import LLM, OpenAI
from llama_index.memory import ChatMemoryBuffer
from llama_index.postprocessor import (PrevNextNodePostprocessor,
                                       SimilarityPostprocessor)
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.retrievers import BaseRetriever, RecursiveRetriever
from llama_index.schema import NodeWithScore
from llama_index.storage.chat_store import SimpleChatStore
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

load_dotenv()


def restore_index(embed_dim: int) -> VectorStoreIndex:
    uri = make_url(connection_string)
    vector_store = PGVectorStore.from_params(
        connection_string=connection_string, port=str(uri.port), embed_dim=embed_dim
    )

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index


def create_retrievers(
    index: VectorStoreIndex, similarity_top_k: int
) -> List[BaseRetriever]:
    retriever = index.as_retriever(
        similarity_top_k=similarity_top_k,
        vector_store_kwargs={
            "ivfflat_probes": 10,  # higher is better for recall, lower is better for speed. Default = 1
            "hnsw_ef_search": 300,  # Specify the size of the dynamic candidate list for search. Default = 40
        },
    )

    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": retriever},
        # verbose=True,
    )

    # TODO: Add more retrievers if needed

    return [retriever, recursive_retriever]


class FusionRetriever(BaseRetriever):
    def __init__(
        self,
        llm: LLM,
        retrievers: List[BaseRetriever],
        similarity_top_k: int,
        query_gen_prompt_template: str,
        reranking_impact: float = 60.0,
        num_queries: int = 4,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._callback_manager = callback_manager
        self._llm = llm
        self._num_queries = num_queries
        self._reranking_impact = reranking_impact
        self._similarity_top_k = similarity_top_k
        self._retrievers = retrievers
        self._query_gen_prompt_template = query_gen_prompt_template

        super().__init__(callback_manager)

    # Derive abstract method
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries = self._generate_queries(llm, query_bundle)
        results = self._run_queries(queries)
        fused_results = self._fuse_results(results)

        return fused_results

    def _generate_queries(self, llm: LLM, query_bundle: QueryBundle) -> List[str]:
        query_gen_prompt = PromptTemplate(query_gen_prompt_template)
        prompt = query_gen_prompt.format(
            num_queries=self._num_queries, query=query_bundle
        )
        response = llm.complete(prompt)
        queries = response.text.split("\n")
        print("Queries generated:", queries)

        return queries

    def _run_queries(self, queries: List[str]) -> Dict[str, List[NodeWithScore]]:
        results = []
        for query in queries:
            for retriever in self._retrievers:
                # TODO: This one should be promisable
                results.append(retriever.retrieve(query))

        results_dict: Dict[str, List[NodeWithScore]] = {}
        for query, query_result in zip(queries, results):
            results_dict[query] = query_result

        return results_dict

    def _fuse_results(
        self,
        results_dict: Dict[str, List[NodeWithScore]],
    ) -> List[NodeWithScore]:
        fused_scores: Dict[str, float] = {}
        text_to_node: Dict[str, NodeWithScore] = {}

        # compute reciprocal rank scores
        for node_with_scores in results_dict.values():
            for rank, node_with_score in enumerate(
                sorted(node_with_scores, key=lambda x: x.score or 0.0, reverse=True)
            ):
                text = node_with_score.get_content()
                text_to_node[text] = node_with_score

                if text not in fused_scores:
                    fused_scores[text] = 0.0
                fused_scores[text] += 1.0 / (rank + self._reranking_impact)

        # sort results
        reranked_results = dict(
            sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        )

        # add node scores
        reranked_nodes: List[NodeWithScore] = []
        for text, score in reranked_results.items():
            reranked_nodes.append(text_to_node[text])
            reranked_nodes[-1].score = score

        return reranked_nodes[: self._similarity_top_k]


def main() -> None:
    index = restore_index(embed_dim)
    retrievers = create_retrievers(index, similarity_top_k)

    fusion_retriever = FusionRetriever(
        llm=llm,
        retrievers=retrievers,
        similarity_top_k=similarity_top_k,
        query_gen_prompt_template=query_gen_prompt_template,
        reranking_impact=reranking_impact,
        num_queries=num_queries,
        callback_manager=callback_manager,
    )

    chat_store = SimpleChatStore()
    memory = ChatMemoryBuffer.from_defaults(
        # token_limit=3000,
        chat_store=chat_store,
        chat_store_key="user-1",
    )

    # system_prompt = PromptTemplate("")

    # context_prompt = PromptTemplate("")

    # condense_prompt = PromptTemplate("")

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=fusion_retriever,
        memory=memory,
        # system_prompt=system_prompt,
        # context_prompt=context_prompt,
        # condense_prompt=condense_prompt,
        verbose=True,
    )

    while True:
        query = input("YOU: ")
        if "end" == query.lower():
            break

        print("GPT:")
        response = chat_engine.chat(query)
        print(response)

        chat_store.persist(persist_path="chat-history/chat-store.json")


if __name__ == "__main__":
    connection_string = os.getenv("CONNECTION_STRING")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GOOGLE_API_KEY")

    # llm = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.0, api_key=openai_api_key)
    llm = OpenAI(model="gpt-4-1106-preview", temperature=0.0, api_key=openai_api_key)
    embed_model = GeminiEmbedding(api_key=gemini_api_key)
    callback_manager = CallbackManager([LlamaDebugHandler(print_trace_on_end=True)])

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        callback_manager=callback_manager,
    )
    set_global_service_context(service_context)

    num_queries = 2
    similarity_top_k = 6
    reranking_impact = 60.0
    embed_dim = 768  # INFO: 1536 for OpenAI Embedding model
    query_gen_prompt_template = """
    You are a helpful assistant that generates multiple search queries based on a single input query.
    Generate {num_queries} search queries, one on each line related to the following input query:
    Query: {query}
    Queries:
    """

    main()
