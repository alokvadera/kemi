"""LangChain memory adapter for kemi."""

from typing import Any

from kemi import Memory


class KemiMemory:
    """LangChain memory adapter using kemi as the backend.

    Extends chat memory to store conversations in kemi and retrieve
    relevant context for each new conversation turn.

    Usage:
        from kemi import Memory
        from kemi.integrations.langchain import KemiMemory

        memory = Memory()
        chat_memory = KemiMemory(user_id="user123", memory=memory)

        # Use with LangChain agents
        from langchain.agents import AgentExecutor
        from langchain.agents import create_openai_functions_agent
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(ChatOpenAI(), prompt, tools=[])
        agent_executor = AgentExecutor(agent=agent, tools=[], memory=chat_memory)
    """

    def __init__(
        self,
        user_id: str,
        memory: Memory,
        query_key: str = "input",
        output_key: str = "output",
    ):
        """Initialize KemiMemory.

        Args:
            user_id: Unique identifier for the user.
            memory: Kemia Memory instance.
            query_key: Key in input dict to use as the query. Default "input".
            output_key: Key in output dict to use as the response. Default "output".
        """
        self.user_id = user_id
        self.kemi_memory = memory
        self.query_key = query_key
        self.output_key = output_key

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Save the human input to kemi memory.

        Args:
            inputs: Dict with the human input (key is self.query_key).
            outputs: Dict with the AI response (key is self.output_key).
        """
        human_input = inputs.get(self.query_key, "")
        if human_input:
            self.kemi_memory.remember(
                user_id=self.user_id,
                content=human_input,
            )

    def load_memory_variables(self, inputs: dict) -> dict[str, Any]:
        """Load relevant context from kemi memory.

        Args:
            inputs: Dict with the current query (key is self.query_key).

        Returns:
            Dict with "history" key containing the context string.
        """
        query = inputs.get(self.query_key, "")
        if query:
            context = self.kemi_memory.context_block(
                user_id=self.user_id,
                query=query,
            )
        else:
            context = ""

        return {"history": context}

    def clear(self) -> None:
        """Clear all memories for this user."""
        self.kemi_memory.forget(self.user_id)
