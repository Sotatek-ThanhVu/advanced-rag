from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from llama_index import (PromptTemplate, QueryBundle, ServiceContext,
                         StorageContext, get_response_synthesizer,
                         load_index_from_storage)
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings import BaseEmbedding, GeminiEmbedding
from llama_index.llms import LLM, OpenAI
from llama_index.llms.utils import LLMType
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import ResponseMode
from llama_index.retrievers import (BaseRetriever, BM25Retriever,
                                    KnowledgeGraphRAGRetriever,
                                    VectorIndexRetriever)
from llama_index.retrievers.auto_merging_retriever import AutoMergingRetriever
from llama_index.schema import NodeWithScore

load_dotenv()


class FusionRetriever(BaseRetriever):
    """
    Ensemble retriever with fusion.

    Args:
        - llm (LLM): Large Language Model for generating new queries
        - retrievers (List[BaseRetriever]): List of retrievers
        - num_queries (int): Number of queries for generating
        - similarity_top_k (int): Number of similar vector
        - k (float): control the impact of outlier rankings
        - callback_manager (CallbackManager)

    """

    def __init__(
        self,
        llm: LLM,
        retrievers: List[BaseRetriever],
        num_queries: int = 4,
        similarity_top_k: int = 2,
        k: float = 60.0,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._llm = llm
        self._retrievers = retrievers
        self._num_queries = num_queries
        self._similarity_top_k = similarity_top_k
        self._k = k
        self._query_gen_prompt_str = """
            You are a helpful assistant that generates multiple search queries based on a single input query.
            Generate {num_queries} search queries, one on each line related to the following input query:
            Query: {query}
            Queries:
        """
        super().__init__(callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve"""

        queries = self._generate_queries(query_bundle)
        results = self._run_queries(queries)
        fused_results = self._fuse_results(results)

        return fused_results

    def _generate_queries(self, query_bundle: QueryBundle) -> List[str]:
        """
        Generate new queries base on provided input query
        """

        query_gen_prompt = PromptTemplate(self._query_gen_prompt_str)
        prompt = query_gen_prompt.format(
            num_queries=self._num_queries, query=query_bundle
        )

        response = self._llm.complete(prompt)
        queries = response.text.split("\n")
        print("Generated new queries:")
        print(queries, sep="\n")

        return queries

    def _run_queries(
        self, queries: List[str]
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        """
        Run queries against retrievers
        """

        results = []
        for query in queries:
            for retriever in self._retrievers:
                results.append(retriever.retrieve(query))

        results_dict: Dict[Tuple[str, int], List[NodeWithScore]] = {}
        for i, (query, query_result) in enumerate(zip(queries, results)):
            results_dict[(query, i)] = query_result

        return results_dict

    def _fuse_results(
        self,
        results_dict: Dict[Tuple[str, int], List[NodeWithScore]],
    ) -> List[NodeWithScore]:
        """
        Combine the results from serveral retrievers into one and re-ranking
        """

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
                fused_scores[text] += 1.0 / (rank + self._k)

        # sort results
        reranked_results = dict(
            sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        )

        # adjust node scores
        reranked_nodes: List[NodeWithScore] = []
        for text, score in reranked_results.items():
            reranked_nodes.append(text_to_node[text])
            reranked_nodes[-1].score = score

        return reranked_nodes[: self._similarity_top_k]


def display_nodes(nodes: List[NodeWithScore]):
    for node in nodes:
        print("\n" + "-" * 50 + "\n")
        print("Node ID:", node.id_)
        print("Node score:", node.score)
        print("Node content:", node.text)
        print("\n" + "-" * 50 + "\n")


def build_context(
    llm: LLMType | None,
    embed_model: BaseEmbedding | None,
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
    auto_merging_retriever = AutoMergingRetriever(
        base_retriever,
        storage_context=storage_context,
        verbose=True,
        # callback_manager=callback_manager,
    )
    bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=10)
    vector_index_retriever = VectorIndexRetriever(
        index=index, similarity_top_k=10, callback_manager=callback_manager
    )
    knowledge_graph_rag_retriever = KnowledgeGraphRAGRetriever(
        service_context=service_context,
        storage_context=storage_context,
        llm=llm,
        verbose=True,
        callback_manager=callback_manager,
    )
    retrievers = [
        base_retriever,
        bm25_retriever,
        auto_merging_retriever,
        vector_index_retriever,
        knowledge_graph_rag_retriever,
    ]
    fusion_retriever = FusionRetriever(
        llm,
        retrievers=retrievers,
        similarity_top_k=10,
        callback_manager=callback_manager,
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=fusion_retriever,
        response_synthesizer=get_response_synthesizer(
            service_context=service_context,
            callback_manager=callback_manager,
            streaming=True,
            verbose=True,
        ),
        # response_mode=ResponseMode.NO_TEXT,
        streaming=True,
    )
    response = query_engine.query(query)
    print(response.get_formatted_sources())
    response.print_response_stream()


if __name__ == "__main__":
    # query = "What confenreces are held in Hawaii? Return in array of id only."
    query = "Radiology conference in exotic places, >3 days and at least 20 points. Return in array of id only."

    llm = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.0)
    embed_model = GeminiEmbedding()
    callback_manager = CallbackManager([LlamaDebugHandler(print_trace_on_end=True)])

    main()
