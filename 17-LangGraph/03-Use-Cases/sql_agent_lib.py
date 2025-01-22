import uuid
from typing import Callable, List
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph


def random_uuid():
    """
    Generate a random UUID and return it as a string.
    """
    return str(uuid.uuid4())


def stream_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str] = [],
    callback: Callable = None,
):
    """
    Stream the execution results of a LangGraph instance in real-time.

    Args:
        graph (CompiledStateGraph): The compiled LangGraph object to execute.
        inputs (dict): A dictionary of input values to pass to the graph.
        config (RunnableConfig): Configuration for execution.
        node_names (List[str], optional): A list of node names to output. Defaults to an empty list, meaning all nodes are included.
        callback (Callable, optional): A callback function to process each chunk. Defaults to None.
            The callback function should accept a dictionary in the format {"node": str, "content": str}.

    Returns:
        None: The function streams results to the output without returning a value.
    """
    prev_node = ""
    for chunk_msg, metadata in graph.stream(inputs, config, stream_mode="messages"):
        curr_node = metadata["langgraph_node"]

        # Process only if node_names is empty or the current node is in node_names
        if not node_names or curr_node in node_names:
            # Execute the callback function if provided
            if callback:
                callback({"node": curr_node, "content": chunk_msg.content})
            else:
                # Print a separator only when the node changes
                if curr_node != prev_node:
                    print("\n" + "=" * 50)
                    print(f"🔄 Node: \033[1;36m{curr_node}\033[0m 🔄")
                    print("- " * 25)
                print(chunk_msg.content, end="", flush=True)

            prev_node = curr_node


def invoke_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str] = [],
    callback: Callable = None,
):
    """
    Stream and display the execution results of a LangGraph application in a formatted manner.

    Args:
        graph (CompiledStateGraph): The compiled LangGraph object to execute.
        inputs (dict): A dictionary of input values to pass to the graph.
        config (RunnableConfig): Configuration for execution.
        node_names (List[str], optional): A list of node names to include in the output. Defaults to an empty list, meaning all nodes are included.
        callback (Callable, optional): A callback function to process each chunk. Defaults to None.
            The callback function should accept a dictionary in the format {"node": str, "content": str}.

    Returns:
        None: The function streams results to the output without returning a value.
    """

    def format_namespace(namespace):
        """
        Format the namespace string to display the graph hierarchy.
        """
        return namespace[-1].split(":")[0] if len(namespace) > 0 else "root graph"

    # Include subgraph outputs by setting subgraphs=True
    for namespace, chunk in graph.stream(
        inputs, config, stream_mode="updates", subgraphs=True
    ):
        for node_name, node_chunk in chunk.items():
            # Skip nodes not in node_names if node_names is specified
            if len(node_names) > 0 and node_name not in node_names:
                continue

            # Execute the callback function if provided
            if callback is not None:
                callback({"node": node_name, "content": node_chunk})
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

                # Print the chunk data for the node
                if isinstance(node_chunk, dict):
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
                        else:
                            print(f"\033[1;32m{k}\033[0m:\n{v}")
                else:
                    if node_chunk is not None:
                        for item in node_chunk:
                            print(item)
                print("=" * 50)


def count_consecutive_errors(messages: list) -> int:
    """
    Count the number of consecutive error messages starting from the end of the list.

    Args:
        messages (list): A list of messages, where each message should have a `content` attribute.

    Returns:
        int: The number of consecutive messages starting with "Error:".
    """
    count = 0

    # Traverse the list in reverse order
    for msg in reversed(messages):
        # Check if the message content starts with "Error:"
        if msg.content.strip().startswith("Error:"):
            count += 1
        else:
            # Stop counting if the sequence of errors is broken
            break

    return count
