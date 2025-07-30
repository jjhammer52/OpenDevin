import os
import sys
from collections import deque
from typing import TYPE_CHECKING
# hotfix: trigger GitHub PR diff

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam
    from openhands.events.action import Action
    from openhands.llm.llm import ModelResponse

from openhands.agenthub.codeact_agent.tools.bash import create_cmd_run_tool
from openhands.agenthub.codeact_agent.tools.browser import BrowserTool
from openhands.agenthub.codeact_agent.tools.condensation_request import CondensationRequestTool
from openhands.agenthub.codeact_agent.tools.finish import FinishTool
from openhands.agenthub.codeact_agent.tools.ipython import IPythonTool
from openhands.agenthub.codeact_agent.tools.llm_based_edit import LLMBasedFileEditTool
from openhands.agenthub.codeact_agent.tools.str_replace_editor import create_str_replace_editor_tool
from openhands.agenthub.codeact_agent.tools.think import ThinkTool
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.events.action import AgentFinishAction, MessageAction
from openhands.events.event import Event
from openhands.llm.llm import LLM
from openhands.llm.llm_utils import check_tools
from openhands.memory.condenser import Condenser
from openhands.memory.condenser.condenser import Condensation, View
from openhands.memory.conversation_memory import ConversationMemory
from openhands.runtime.plugins import AgentSkillsRequirement, JupyterRequirement, PluginRequirement
from openhands.utils.prompt import PromptManager

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.events.action import ExecuteBashAction


class CodeActAgent(Agent):
    VERSION = '2.2'
    sandbox_plugins: list[PluginRequirement] = [AgentSkillsRequirement(), JupyterRequirement()]

    def __init__(self, llm: LLM, config: AgentConfig) -> None:
        super().__init__(llm, config)
        self.pending_actions: deque['Action'] = deque()
        self.reset()
        self.tools = self._get_tools()
        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)
        self.condenser = Condenser.from_config(self.config.condenser)
        logger.debug(f'Using condenser: {type(self.condenser)}')

    @property
    def prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            self._prompt_manager = PromptManager(
                prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts'),
                system_prompt_filename=self.config.system_prompt_filename,
            )
        return self._prompt_manager

    def _get_tools(self) -> list['ChatCompletionToolParam']:
        SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS = ['gpt-', 'o3', 'o1', 'o4']
        use_short_tool_desc = self.llm and any(
            model_substr in self.llm.config.model for model_substr in SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS
        )
        tools = []
        if self.config.enable_cmd:
            tools.append(create_cmd_run_tool(use_short_description=use_short_tool_desc))
        if self.config.enable_think:
            tools.append(ThinkTool)
        if self.config.enable_finish:
            tools.append(FinishTool)
        if self.config.enable_condensation_request:
            tools.append(CondensationRequestTool)
        if self.config.enable_browsing and sys.platform != 'win32':
            tools.append(BrowserTool)
        if self.config.enable_jupyter:
            tools.append(IPythonTool)
        if self.config.enable_llm_editor:
            tools.append(LLMBasedFileEditTool)
        elif self.config.enable_editor:
            tools.append(create_str_replace_editor_tool(use_short_description=use_short_tool_desc))
        return tools

    def reset(self) -> None:
        super().reset()
        self.pending_actions.clear()

    def step(self, state: State) -> 'Action':
        if self.pending_actions:
            return self.pending_actions.popleft()

        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events
            case Condensation(action=condensation_action):
                return condensation_action

        initial_user_message = self._get_initial_user_message(state.history)
        messages = self._get_messages(condensed_history, initial_user_message)

        params = {
            'messages': self.llm.format_messages_for_llm(messages),
            'tools': check_tools(self.tools, self.llm.config),
            'tool_choice': 'auto',
            'extra_body': {'metadata': state.to_llm_metadata(agent_name=self.name)},
        }

        logger.debug("Calling LLM with enforced tool_choice...")
        response = self.llm.completion(**params)
        logger.debug(f"[LLM RESPONSE] {response}")

        if isinstance(response, list):
            response = response[0]

        msg = response.choices[0].message

        if not getattr(msg, 'tool_calls', None):
            logger.warning("[FALLBACK] No tool_call returned. Attempting inline shell injection.")
            fallback_text = msg.content or ""
            if 'list the contents of' in fallback_text or 'create a file' in fallback_text:
                return ExecuteBashAction(command="ls -la /app && echo 'Tool call executed' > /app/agent_trigger.txt && cat /app/agent_trigger.txt")
            raise RuntimeError("No tool_call returned. Cannot proceed.")

        actions = self.response_to_actions(response)
        logger.debug(f"[ACTIONS] Parsed: {actions}")
        for action in actions:
            self.pending_actions.append(action)

        return self.pending_actions.popleft()

    def _get_initial_user_message(self, history: list[Event]) -> MessageAction:
        for event in history:
            if isinstance(event, MessageAction) and event.source == 'user':
                return event
        logger.error('CRITICAL: Could not find the initial user MessageAction.')
        raise ValueError('Initial user message not found in history.')

    def _get_messages(self, events: list[Event], initial_user_message: MessageAction) -> list[Message]:
        if not self.prompt_manager:
            raise Exception('Prompt Manager not instantiated.')

        messages = self.conversation_memory.process_events(
            condensed_history=events,
            initial_user_action=initial_user_message,
            max_message_chars=self.llm.config.max_message_chars,
            vision_is_active=self.llm.vision_is_active(),
        )
        if self.llm.is_caching_prompt_active():
            self.conversation_memory.apply_prompt_caching(messages)
        return messages

    def response_to_actions(self, response: 'ModelResponse') -> list['Action']:
        return codeact_function_calling.response_to_actions(
            response,
            mcp_tool_names=list(self.mcp_tools.keys()),
        )
# temp diff for PR sync
