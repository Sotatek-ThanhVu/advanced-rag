from typing import List, Optional, Tuple

from llama_index.node_parser import (HierarchicalNodeParser,
                                     SentenceWindowNodeParser, get_leaf_nodes)
from llama_index.readers import Document
from llama_index.schema import BaseNode


class NodeParser:
    """Parsing Documents into Text Chunks (Nodes)

    Args:
        document (Document): document to parse
        show_progress (bool): Show to parsing process

        # Using Auto Merging Retrieval (Parent Document Retriever)

        `HierarchicalNodeParser` will output a hierarchy of nodes, from top-level nodes
        with bigger chunk sizes to child nodes  with smaller chunk sizes, where each
        child node has a parent node with a bigger chunk size.

        chunk_sizes (List[int]): The chunk sizes to use when splitting documents, in order of level.

        OR

        # Using Sentence Window Retrieval

        `SentenceWindowNodeParser` will splits a document into Nodes, with each node being a sentence.
        Each node contains a window from the surrounding sentences in the metadata.

        window_size (str): The number of sentences on each side of a sentence to capture.
        window_metadata_key (str): The metadata key to store the sentence window under.
        original_text_metadata_key (str): The metadata key to store the original sentence in.

    Returns:
        NodeParser: A NodeParser object
    """

    def __init__(
        self,
        document: Document,
        chunk_sizes: Optional[List[int]] = None,
        window_size: Optional[int] = None,
        window_metadata_key: Optional[str] = None,
        original_text_metadata_key: Optional[str] = None,
        show_progress: bool = False,
    ):
        if chunk_sizes:
            self.node_parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=chunk_sizes
            )
        elif window_size and window_metadata_key and original_text_metadata_key:
            self.node_parser = SentenceWindowNodeParser.from_defaults(
                window_size=window_size,
                window_metadata_key=window_metadata_key,
                original_text_metadata_key=original_text_metadata_key,
            )
        else:
            raise Exception("Unsupport node parser type")

        self.nodes = self.node_parser.get_nodes_from_documents(
            documents=[document], show_progress=show_progress
        )
        # for idx, node in enumerate(self.nodes):
        #     node.id_ = f"node-{idx}"

        self.leaf_nodes = get_leaf_nodes(self.nodes)

    def get_nodes(self) -> Tuple[List[BaseNode], List[BaseNode]]:
        return (self.nodes, self.leaf_nodes)
