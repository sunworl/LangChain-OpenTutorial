from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Iterable
from langchain_core.documents import Document


class DocumentManagerInterface(ABC):
    """
    문서 insert/update (upsert, upsert_parallel)
    문서 search by query (search)
    문서 delete by id, delete by filter (delete)
    """

    @abstractmethod
    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """
        Upsert Documents
        """
        pass

    @abstractmethod
    def upsert_parallel(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        workers: int = 10,
        **kwargs: Any
    ) -> None:
        """
        Upsert Documnets in parallel
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 10, **kwargs: Any) -> List[Document]:
        """쿼리를 수행하고 관련 문서를 반환합니다.
        기본 기능: query (문자열) -> 비슷한 문서 k개 반환

        cosine_similarity 써치하는 것 의미 **문제될 경우 이슈제기

        -그외 기능 (추후 확장)
        metatdata search
        이미지 서치할 때 벡터 받는 것
        """
        pass

    @abstractmethod
    def delete(
        self,
        ids: Optional[list[str]] = None,
        filters: Optional[dict] = None,
        **kwargs: Any
    ) -> None:
        """필터를 사용하여 문서를 삭제합니다.

        ids: List of ids to delete. If None, delete all. Default is None.
        filters: Dictionary of filters (querys) to apply. If None, no filters apply.

        """
        pass
