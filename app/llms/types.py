from typing import Literal, TypedDict, Optional, List

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

class LLMResponse(TypedDict):
    content: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    model: str