import json
import os
import random
import uuid
from dataclasses import dataclass
from typing import List, Callable, Literal, Sequence, Optional
from pydantic import BaseModel, Field
from IPython.display import Image, display
from tavily import TavilyClient
from langgraph.graph.state import CompiledStateGraph
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig


class TavilySearchInput(BaseModel):
    """Input for the Tavily tool."""

    query: str = Field(description="검색 쿼리")


def format_search_result(result: dict, include_raw_content: bool = False) -> str:
    """
    Utility functions for formatting search results.

    Args:
        result (dict): 원본 검색 결과

    Returns:
        str: XML 형식으로 포맷팅된 검색 결과
    """
    # 한글 인코딩 처리를 위해 json.dumps() 사용
    title = json.dumps(result["title"], ensure_ascii=False)[1:-1]
    content = json.dumps(result["content"], ensure_ascii=False)[1:-1]
    raw_content = ""
    if (
        include_raw_content
        and "raw_content" in result
        and result["raw_content"] is not None
        and len(result["raw_content"].strip()) > 0
    ):
        raw_content = f"<raw>{result['raw_content']}</raw>"

    return f"<document><title>{title}</title><url>{result['url']}</url><content>{content}</content>{raw_content}</document>"


class TavilySearch(BaseTool):
    """
    Tool that queries the Tavily Search API and gets back json
    """

    name: str = "tavily_web_search"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. [IMPORTANT] Input(query) should be over 5 characters."
    )
    args_schema: type[BaseModel] = TavilySearchInput
    client: TavilyClient = None
    include_domains: list = []
    exclude_domains: list = []
    max_results: int = 3
    topic: Literal["general", "news"] = "general"
    days: int = 3
    search_depth: Literal["basic", "advanced"] = "basic"
    include_answer: bool = False
    include_raw_content: bool = True
    include_images: bool = False
    format_output: bool = False

    def __init__(
        self,
        api_key: Optional[str] = None,
        include_domains: list = [],
        exclude_domains: list = [],
        max_results: int = 3,
        topic: Literal["general", "news"] = "general",
        days: int = 3,
        search_depth: Literal["basic", "advanced"] = "basic",
        include_answer: bool = False,
        include_raw_content: bool = True,
        include_images: bool = False,
        format_output: bool = False,
    ):
        """
        TavilySearch 클래스의 인스턴스를 초기화합니다.

        Args:
            api_key (str): Tavily API 키
            include_domains (list): 검색에 포함할 도메인 목록
            exclude_domains (list): 검색에서 제외할 도메인 목록
            max_results (int): 기본 검색 결과 수
        """
        super().__init__()
        if api_key is None:
            api_key = os.environ.get("TAVILY_API_KEY", None)

        if api_key is None:
            raise ValueError("Tavily API key is not set.")

        self.client = TavilyClient(api_key=api_key)
        self.include_domains = include_domains
        self.exclude_domains = exclude_domains
        self.max_results = max_results
        self.topic = topic
        self.days = days
        self.search_depth = search_depth
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        self.include_images = include_images
        self.format_output = format_output

    def _run(self, query: str) -> str:
        """BaseTool의 _run 메서드 구현"""
        results = self.search(query)
        return results
        # return json.dumps(results, ensure_ascii=False)

    def search(
        self,
        query: str,
        search_depth: Literal["basic", "advanced"] = None,
        topic: Literal["general", "news"] = None,
        days: int = None,
        max_results: int = None,
        include_domains: Sequence[str] = None,
        exclude_domains: Sequence[str] = None,
        include_answer: bool = None,
        include_raw_content: bool = None,
        include_images: bool = None,
        format_output: bool = None,
        **kwargs,
    ) -> list:
        """
        검색을 수행하고 결과를 반환합니다.

        Args:
            query (str): 검색 쿼리
            search_depth (str): 검색 깊이 ("basic" 또는 "advanced")
            topic (str): 검색 주제 ("general" 또는 "news")
            days (int): 검색할 날짜 범위
            max_results (int): 최대 검색 결과 수
            include_domains (list): 검색에 포함할 도메인 목록
            exclude_domains (list): 검색에서 제외할 도메인 목록
            include_answer (bool): 답변 포함 여부
            include_raw_content (bool): 원본 콘텐츠 포함 여부
            include_images (bool): 이미지 포함 여부
            format_output (bool): 결과를 포맷팅할지 여부
            **kwargs: 추가 키워드 인자

        Returns:
            list: 검색 결과 목록
        """
        # 기본값 설정
        params = {
            "query": query,
            "search_depth": search_depth or self.search_depth,
            "topic": topic or self.topic,
            "max_results": max_results or self.max_results,
            "include_domains": include_domains or self.include_domains,
            "exclude_domains": exclude_domains or self.exclude_domains,
            "include_answer": (
                include_answer if include_answer is not None else self.include_answer
            ),
            "include_raw_content": (
                include_raw_content
                if include_raw_content is not None
                else self.include_raw_content
            ),
            "include_images": (
                include_images if include_images is not None else self.include_images
            ),
            **kwargs,
        }

        # days 파라미터 처리
        if days is not None:
            if params["topic"] == "general":
                print(
                    "Warning: days parameter is ignored for 'general' topic search. Set topic parameter to 'news' to use days."
                )
            else:
                params["days"] = days

        # API 호출
        response = self.client.search(**params)

        # 결과 포맷팅
        format_output = (
            format_output if format_output is not None else self.format_output
        )
        if format_output:
            return [
                format_search_result(r, params["include_raw_content"])
                for r in response["results"]
            ]
        else:
            return response["results"]

    def get_search_context(
        self,
        query: str,
        search_depth: Literal["basic", "advanced"] = "basic",
        topic: Literal["general", "news"] = "general",
        days: int = 3,
        max_results: int = 5,
        include_domains: Sequence[str] = None,
        exclude_domains: Sequence[str] = None,
        max_tokens: int = 4000,
        format_output: bool = True,
        **kwargs,
    ) -> str:
        """
        검색 쿼리에 대한 컨텍스트를 가져옵니다. 웹사이트에서 관련 콘텐츠만 가져오는 데 유용하며,
        컨텍스트 추출과 제한을 직접 처리할 필요가 없습니다.

        Args:
            query (str): 검색 쿼리
            search_depth (str): 검색 깊이 ("basic" 또는 "advanced")
            topic (str): 검색 주제 ("general" 또는 "news")
            days (int): 검색할 날짜 범위
            max_results (int): 최대 검색 결과 수
            include_domains (list): 검색에 포함할 도메인 목록
            exclude_domains (list): 검색에서 제외할 도메인 목록
            max_tokens (int): 반환할 최대 토큰 수 (openai 토큰 계산 기준). 기본값은 4000입니다.
            format_output (bool): 결과를 포맷팅할지 여부
            **kwargs: 추가 키워드 인자

        Returns:
            str: 컨텍스트 제한까지의 검색 컨텍스트를 포함하는 JSON 문자열
        """
        response = self.client.search(
            query,
            search_depth=search_depth,
            topic=topic,
            days=days,
            max_results=max_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
            **kwargs,
        )

        sources = response.get("results", [])
        if format_output:
            context = [
                format_search_result(source, include_raw_content=False)
                for source in sources
            ]
        else:
            context = [
                {
                    "url": source["url"],
                    "content": json.dumps(
                        {"title": source["title"], "content": source["content"]},
                        ensure_ascii=False,
                    ),
                }
                for source in sources
            ]

        # max_tokens 처리 로직은 여기에 구현해야 합니다.
        # 현재는 간단히 모든 컨텍스트를 반환합니다.
        return json.dumps(context, ensure_ascii=False)


@dataclass
class NodeStyles:
    default: str = (
        "fill:#45C4B0, fill-opacity:0.3, color:#23260F, stroke:#45C4B0, stroke-width:1px, font-weight:bold, line-height:1.2"  # 기본 색상
    )
    first: str = (
        "fill:#45C4B0, fill-opacity:0.1, color:#23260F, stroke:#45C4B0, stroke-width:1px, font-weight:normal, font-style:italic, stroke-dasharray:2,2"  # 점선 테두리
    )
    last: str = (
        "fill:#45C4B0, fill-opacity:1, color:#000000, stroke:#45C4B0, stroke-width:1px, font-weight:normal, font-style:italic, stroke-dasharray:2,2"  # 점선 테두리
    )


def visualize_graph(graph, xray=False):
    """
    CompiledStateGraph 객체를 시각화하여 표시합니다.

    이 함수는 주어진 그래프 객체가 CompiledStateGraph 인스턴스인 경우
    해당 그래프를 Mermaid 형식의 PNG 이미지로 변환하여 표시합니다.

    Args:
        graph: 시각화할 그래프 객체. CompiledStateGraph 인스턴스여야 합니다.

    Returns:
        None

    Raises:
        Exception: 그래프 시각화 과정에서 오류가 발생한 경우 예외를 출력합니다.
    """
    try:
        # 그래프 시각화
        if isinstance(graph, CompiledStateGraph):
            display(
                Image(
                    graph.get_graph(xray=xray).draw_mermaid_png(
                        background_color="white",
                        node_colors=NodeStyles(),
                    )
                )
            )
    except Exception as e:
        print(f"[ERROR] Visualize Graph Error: {e}")


def generate_random_hash():
    return f"{random.randint(0, 0xffffff):06x}"


def random_uuid():
    return str(uuid.uuid4())


def invoke_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str] = [],
    callback: Callable = None,
):
    """
    LangGraph 앱의 실행 결과를 예쁘게 스트리밍하여 출력하는 함수입니다.

    Args:
        graph (CompiledStateGraph): 실행할 컴파일된 LangGraph 객체
        inputs (dict): 그래프에 전달할 입력값 딕셔너리
        config (RunnableConfig): 실행 설정
        node_names (List[str], optional): 출력할 노드 이름 목록. 기본값은 빈 리스트
        callback (Callable, optional): 각 청크 처리를 위한 콜백 함수. 기본값은 None
            콜백 함수는 {"node": str, "content": str} 형태의 딕셔너리를 인자로 받습니다.

    Returns:
        None: 함수는 스트리밍 결과를 출력만 하고 반환값은 없습니다.
    """

    def format_namespace(namespace):
        return namespace[-1].split(":")[0] if len(namespace) > 0 else "root graph"

    # subgraphs=True 를 통해 서브그래프의 출력도 포함
    for namespace, chunk in graph.stream(
        inputs, config, stream_mode="updates", subgraphs=True
    ):
        for node_name, node_chunk in chunk.items():
            # node_names가 비어있지 않은 경우에만 필터링
            if len(node_names) > 0 and node_name not in node_names:
                continue

            # 콜백 함수가 있는 경우 실행
            if callback is not None:
                callback({"node": node_name, "content": node_chunk})
            # 콜백이 없는 경우 기본 출력
            else:
                print("\n" + "=" * 50)
                formatted_namespace = format_namespace(namespace)
                if formatted_namespace == "root graph":
                    print(f"🔄 Node: \033[1;36m{node_name}\033[0m 🔄")
                else:
                    print(
                        f"🔄 Node: \033[1;36m{node_name}\033[0m in [\033[1;33m{formatted_namespace}\033[0m] 🔄"
                    )
                print("- " * 25)

                # 노드의 청크 데이터 출력
                for k, v in node_chunk.items():
                    if isinstance(v, BaseMessage):
                        v.pretty_print()
                    elif isinstance(v, list):
                        for list_item in v:
                            if isinstance(list_item, BaseMessage):
                                list_item.pretty_print()
                            else:
                                print(list_item)
                    elif isinstance(v, dict):
                        for node_chunk_key, node_chunk_value in node_chunk.items():
                            print(f"{node_chunk_key}:\n{node_chunk_value}")
                print("=" * 50)
