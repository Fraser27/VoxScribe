"""
Strands Agents client for voice agent conversations
Provides model-driven AI agent with conversation memory and multi-provider support
"""
import os
from typing import AsyncGenerator, List, Dict, Optional
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.session import FileSessionStore
from config import (
    AWS_REGION,
    BEDROCK_MODEL_ID,
    SYSTEM_PROMPT,
    MEMORY_STORAGE_PATH,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TOP_P,
)


class StrandsClient:
    """
    Strands Agents client for building conversational AI agents
    Supports multiple LLM providers and automatic session management
    """
    
    def __init__(self):
        # Ensure memory storage directory exists
        os.makedirs(MEMORY_STORAGE_PATH, exist_ok=True)
        
        # Initialize session store for persistent conversation memory
        self.session_store = FileSessionStore(storage_path=MEMORY_STORAGE_PATH)
        
        # Initialize Bedrock model
        self.model = BedrockModel(
            model_id=BEDROCK_MODEL_ID,
            region=AWS_REGION,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            top_p=LLM_TOP_P,
        )
        
        self.system_prompt = SYSTEM_PROMPT
    
    def create_agent(self, session_id: str, tools: Optional[List] = None) -> Agent:
        """
        Create a Strands agent for a specific session
        
        Args:
            session_id: Unique session identifier
            tools: Optional list of tools for the agent
        
        Returns:
            Configured Strands Agent instance
        """
        agent = Agent(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=tools or [],
            session_id=session_id,
            session_store=self.session_store,
        )
        return agent
    
    async def stream_response(
        self,
        session_id: str,
        user_message: str,
        tools: Optional[List] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from Strands agent with automatic conversation memory
        
        Args:
            session_id: Session identifier for conversation continuity
            user_message: Current user message
            tools: Optional list of tools for the agent
        
        Yields:
            Text chunks from the agent
        """
        try:
            # Create agent with session management
            agent = self.create_agent(session_id, tools)
            
            # Stream response from agent
            # Strands automatically manages conversation history via session_store
            async for chunk in agent.stream(user_message):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                elif isinstance(chunk, str):
                    yield chunk
        
        except Exception as e:
            error_msg = f"Strands agent error: {str(e)}"
            print(f"Error in stream_response: {error_msg}")
            yield f"I apologize, but I encountered an error: {error_msg}"
    
    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Retrieve conversation history for a session
        
        Args:
            session_id: Session identifier
        
        Returns:
            List of conversation messages
        """
        try:
            session = self.session_store.load(session_id)
            if session and hasattr(session, 'messages'):
                return [
                    {"role": msg.role, "content": msg.content}
                    for msg in session.messages
                ]
            return []
        except Exception as e:
            print(f"Error retrieving session history: {e}")
            return []
    
    def clear_session(self, session_id: str):
        """
        Clear conversation history for a session
        
        Args:
            session_id: Session identifier
        """
        try:
            self.session_store.delete(session_id)
        except Exception as e:
            print(f"Error clearing session: {e}")
