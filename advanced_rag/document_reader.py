from typing import List, Optional

from llama_index.readers import Document, SimpleDirectoryReader
from llama_index.readers.database import DatabaseReader


class DocumentReader:
    """Loading Documents from source

    This is the first step for data ingestion in LlamaIndex.


    Args:
        show_progress (bool): Show the reading process (only for SimpleDirectoryReader)

        input_dir (Optional[str]): Path to the directory.
        input_files (Optional[List[str]]): List of file paths to read

        OR

        connection_string (Optional[str]): uri of the database connection.
        query (Optional[str]): Query parameter to filter tables and rows.

    Returns:
        DocumentReader: A DocumentReader object
    """

    def __init__(
        self,
        input_dir: Optional[str] = None,
        input_files: Optional[List[str]] = None,
        connection_string: Optional[str] = None,
        query: Optional[str] = None,
        show_progress: bool = False,
    ) -> None:
        self.show_progress = show_progress
        if input_dir or input_files:
            self.documents = self._directory_reader(
                input_dir=input_dir, input_files=input_files
            )
        elif connection_string and query:
            self.documents = self._database_reader(
                connection_string=connection_string, query=query
            )
        else:
            raise Exception("Unsupport document reader type")

    def _directory_reader(
        self,
        input_dir: str | None,
        input_files: List[str] | None,
    ) -> List[Document]:
        loader = SimpleDirectoryReader(input_dir=input_dir, input_files=input_files)
        return loader.load_data(show_progress=self.show_progress)

    def _database_reader(
        self, connection_string: str | None, query: str | None
    ) -> List[Document]:
        if connection_string is None:
            raise Exception("Connection string must be set")

        if query is None:
            raise Exception("Query string must be set")

        loader = DatabaseReader(uri=connection_string)
        return loader.load_data(query=query)

    def get_document(self) -> Document:
        documents = Document(
            text="\n\n".join([doc.get_content() for doc in self.documents])
        )
        return documents
