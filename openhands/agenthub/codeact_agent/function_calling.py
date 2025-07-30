"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import json

from litellm import ModelResponse

from openhands.agenthub.codeact_agent.tools import (
    BrowserTool,
    CondensationRequestTool,
    FinishTool,
    IPythonTool,
    LLMBasedFileEditTool,
    ThinkTool,
    create_cmd_run_tool,
    create_str_replace_editor_tool,
)
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    AgentDelegateAction,
    AgentFinishAction,
    AgentThinkAction,
    BrowseInteractiveAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    IPythonRunCellAction,
    MessageAction,
)
from openhands.events.action.agent import CondensationRequestAction
from openhands.events.action.mcp import MCPAction
from openhands.events.event import FileEditSource, FileReadSource
from openhands.events.tool import ToolCallMetadata


def combine_thought(action: Action, thought: str) -> Action:
    if not hasattr(action, "thought"):
        return action
    if thought and action.thought:
        action.thought = f"{thought}\n{action.thought}"
    elif thought:
        action.thought = thought
    return action


def response_to_actions(
    response: ModelResponse, mcp_tool_names: list[str] | None = None
) -> list[Action]:
    actions: list[Action] = []
    assert len(response.choices) == 1, "Only one choice is supported for now"
    choice = response.choices[0]
    assistant_msg = choice.message

    if hasattr(assistant_msg, "tool_calls") and assistant_msg.tool_calls:
        # Patch: handle multi-tool calls by trimming to one
        tool_calls = assistant_msg.tool_calls
        if len(tool_calls) > 1:
            logger.warning(f"[PATCH] Multiple tool calls detected: trimming to first.")
            tool_calls = [tool_calls[0]]

        thought = ""
        if isinstance(assistant_msg.content, str):
            thought = assistant_msg.content
        elif isinstance(assistant_msg.content, list):
            for msg in assistant_msg.content:
                if msg["type"] == "text":
                    thought += msg["text"]

        for i, tool_call in enumerate(tool_calls):
            logger.debug(f"Tool call in function_calling.py: {tool_call}")
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.decoder.JSONDecodeError as e:
                raise FunctionCallValidationError(
                    f"Failed to parse tool call arguments: {tool_call.function.arguments}"
                ) from e

            name = tool_call.function.name
            if name == create_cmd_run_tool()["function"]["name"]:
                is_input = arguments.get("is_input", "false") == "true"
                action = CmdRunAction(command=arguments["command"], is_input=is_input)
                if "timeout" in arguments:
                    try:
                        action.set_hard_timeout(float(arguments["timeout"]))
                    except ValueError as e:
                        raise FunctionCallValidationError(
                            f"Invalid float in 'timeout': {arguments['timeout']}"
                        ) from e
            elif name == IPythonTool["function"]["name"]:
                action = IPythonRunCellAction(code=arguments["code"])
            elif name == "delegate_to_browsing_agent":
                action = AgentDelegateAction(agent="BrowsingAgent", inputs=arguments)
            elif name == FinishTool["function"]["name"]:
                action = AgentFinishAction(
                    final_thought=arguments.get("message", ""),
                    task_completed=arguments.get("task_completed"),
                )
            elif name == LLMBasedFileEditTool["function"]["name"]:
                action = FileEditAction(
                    path=arguments["path"],
                    content=arguments["content"],
                    start=arguments.get("start", 1),
                    end=arguments.get("end", -1),
                    impl_source=arguments.get(
                        "impl_source", FileEditSource.LLM_BASED_EDIT
                    ),
                )
            elif name == create_str_replace_editor_tool()["function"]["name"]:
                path = arguments["path"]
                command = arguments["command"]
                other_kwargs = {
                    k: v for k, v in arguments.items() if k not in ["command", "path"]
                }
                if command == "view":
                    action = FileReadAction(
                        path=path,
                        impl_source=FileReadSource.OH_ACI,
                        view_range=other_kwargs.get("view_range"),
                    )
                else:
                    other_kwargs.pop("view_range", None)
                    tool = create_str_replace_editor_tool()
                    valid_keys = set(tool["function"]["parameters"]["properties"].keys())
                    valid_args = {
                        k: v for k, v in other_kwargs.items() if k in valid_keys
                    }
                    action = FileEditAction(
                        path=path,
                        command=command,
                        impl_source=FileEditSource.OH_ACI,
                        **valid_args,
                    )
            elif name == ThinkTool["function"]["name"]:
                action = AgentThinkAction(thought=arguments.get("thought", ""))
            elif name == CondensationRequestTool["function"]["name"]:
                action = CondensationRequestAction()
            elif name == BrowserTool["function"]["name"]:
                action = BrowseInteractiveAction(browser_actions=arguments["code"])
            elif mcp_tool_names and name in mcp_tool_names:
                action = MCPAction(name=name, arguments=arguments)
            else:
                raise FunctionCallNotExistsError(
                    f"Tool {name} not registered. Args: {arguments}"
                )

            if i == 0:
                action = combine_thought(action, thought)
            action.tool_call_metadata = ToolCallMetadata(
                tool_call_id=tool_call.id,
                function_name=name,
                model_response=response,
                total_calls_in_response=len(assistant_msg.tool_calls),
            )
            actions.append(action)
    else:
        actions.append(
            MessageAction(
                content=str(assistant_msg.content) if assistant_msg.content else "",
                wait_for_response=True,
            )
        )

    for action in actions:
        action.response_id = response.id

    assert len(actions) >= 1
    return actions
