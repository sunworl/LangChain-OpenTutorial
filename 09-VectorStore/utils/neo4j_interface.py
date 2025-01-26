import neo4j
from vectorstore_interface import DocumentManagerInterface
from langchain_core.documents import Document
from typing import List, Union, Dict, Any, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from hashlib import md5
import os, time

METRIC = {
    "cosine": "COSINE",
    "euclidean": "EUCLIDEAN",
}


class Neo4jDBManager(DocumentManagerInterface):
    def __init__(self, client, index_name, embedding):
        self.index_name = index_name
        self.client = client
        self.embedding = embedding
        self.get_index_info()

    def check_neo4j_version(self):
        db_data = self.client.execute_query("CALL dbms.components()")
        version = db_data[0][0]["versions"][0]

        if "aura" in version:
            version_tuple = tuple(map(int, version.split("-")[0].split("."))) + (0,)
        else:
            version_tuple = tuple(map(int, version.split(".")))

        target_version = (5, 11, 0)

        if version_tuple < target_version:
            raise ValueError(
                "Version index is only supported in Neo4j version 5.11 or greater"
            )

        if version_tuple >= (5, 23, 0):
            return True
        else:
            return False

    def get_index_info(self):
        info_query = f"SHOW INDEX YIELD * WHERE name='{self.index_name}' RETURN labelsOrTypes, properties"
        info = self.client.execute_query(info_query).records[0]
        self.node_label = info["labelsOrTypes"][0]
        self.embedding_node_property = info["properties"][0]
        self.text_node_property = "text"

    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Upsert documents into the Neo4j vectorstore

        Args:
        - texts: List of text. If ids is not provided, will create id based on the lowered text.
        - metadats: Optional. List of metadata. Default is None.
        - ids: Optional. List of id.

        Return:
        ids of upserted documents.
        """
        if ids is not None:
            assert len(ids) == len(
                texts
            ), "The length of ids and texts must be the same."
