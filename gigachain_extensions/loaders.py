import json
from typing import Iterator, List, Mapping, Optional, Sequence, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class HuggingFaceDatasetLoader(BaseLoader):
    """Load from `Hugging Face Hub` datasets."""

    def __init__(
        self,
        path: str,
        page_content_column: str = "text",
        keep_in_memory: Optional[bool] = None,
    ):
        """Initialize the HuggingFaceDatasetLoader.

        Args:
            path: Path or name of the dataset.
            page_content_column: Page content column name. Default is "text".
            name: Name of the dataset configuration.
            data_dir: Data directory of the dataset configuration.
            data_files: Path(s) to source data file(s).
            cache_dir: Directory to read/write data.
            keep_in_memory: Whether to copy the dataset in-memory.
            save_infos: Save the dataset information (checksums/size/splits/...).
              Default is False.
            use_auth_token: Bearer token for remote files on the Dataset Hub.
            num_proc: Number of processes.
        """
        self.path = path
        self.page_content_column = page_content_column
        self.keep_in_memory = keep_in_memory

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Load documents lazily."""
        try:
            from datasets import load_from_disk
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "Could not import datasets python package. "
                "Please install it with `pip install datasets`."
            )

        dataset: Dataset = load_from_disk(
            dataset_path=self.path,
            keep_in_memory=self.keep_in_memory,
        )

        yield from (
            Document(
                page_content=self.parse_obj(row.pop(self.page_content_column)),
                metadata=row,
            )
            for row in dataset
        )

    def load(self) -> List[Document]:
        """Load documents."""
        return list(self.lazy_load())

    def parse_obj(self, page_content: str) -> str:
        return page_content
