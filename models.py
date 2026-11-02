from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


class ContentPart(BaseModel):
    """Content part for multimodal messages (OpenAI format)."""
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[ContentPart]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    @model_validator(mode='after')
    def normalize_content(self):
        """Convert array content to string for Claude Code compatibility."""
        if isinstance(self.content, list):
            # Extract text from content parts and concatenate
            text_parts = []
            for part in self.content:
                if isinstance(part, ContentPart) and part.type == "text":
                    text_parts.append(part.text)
                elif isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            
            # Join all text parts with newlines
            self.content = "\n".join(text_parts) if text_parts else ""
            
        return self


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    type: Literal["function"]
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[Literal["auto", "none", "required"], Dict[str, Any]]] = None
    parallel_tool_calls: bool = True
    effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = None
    reasoning_effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = None
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    session_id: Optional[str] = Field(default=None, description="Optional session ID for conversation continuity")
    prompt_cache_key: Optional[str] = Field(default=None, description="OpenAI cache key; used as session_id when that is absent")
    enable_tools: Optional[bool] = Field(default=False, description="Enable Claude Code tools; with Antigravity use the tools field for program functions")

    @model_validator(mode='after')
    def validate_effort(self):
        if self.effort and self.reasoning_effort and self.effort != self.reasoning_effort:
            raise ValueError("effort and reasoning_effort must match when both are provided")
        return self

    @property
    def agy_effort(self) -> Optional[str]:
        return self.effort or self.reasoning_effort
    
    @field_validator('n')
    @classmethod
    def validate_n(cls, v):
        if v > 1:
            raise ValueError("Claude Code SDK does not support multiple choices (n > 1). Only single response generation is supported.")
        return v
    
    def log_unsupported_parameters(self):
        """Log warnings for parameters that are not supported by Claude Code SDK."""
        warnings = []
        
        if self.temperature != 1.0:
            warnings.append(f"temperature={self.temperature} is not supported by Claude Code SDK and will be ignored")
        
        if self.top_p != 1.0:
            warnings.append(f"top_p={self.top_p} is not supported by Claude Code SDK and will be ignored")
            
        if self.max_tokens is not None:
            warnings.append(f"max_tokens={self.max_tokens} is not supported by Claude Code SDK and will be ignored. Consider using max_turns to limit conversation length")
        
        if self.presence_penalty != 0:
            warnings.append(f"presence_penalty={self.presence_penalty} is not supported by Claude Code SDK and will be ignored")
            
        if self.frequency_penalty != 0:
            warnings.append(f"frequency_penalty={self.frequency_penalty} is not supported by Claude Code SDK and will be ignored")
            
        if self.logit_bias:
            warnings.append(f"logit_bias is not supported by Claude Code SDK and will be ignored")
            
        if self.stop:
            warnings.append(f"stop sequences are not supported by Claude Code SDK and will be ignored")
        
        for warning in warnings:
            logger.warning(f"OpenAI API compatibility: {warning}")
    
    def to_claude_options(self) -> Dict[str, Any]:
        """Convert OpenAI request parameters to Claude Code SDK options."""
        # Log warnings for unsupported parameters
        self.log_unsupported_parameters()
        
        options = {}
        
        # Direct mappings
        if self.model:
            options['model'] = self.model
            
        # Use user field for session identification if provided
        if self.user:
            # Could be used for analytics/logging or session tracking
            logger.info(f"Request from user: {self.user}")
        
        return options


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls", "null"]] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class StreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls", "null"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str
    choices: List[StreamChoice]
    system_fingerprint: Optional[str] = None


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    last_accessed: datetime
    message_count: int
    expires_at: datetime


class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]
    total: int
