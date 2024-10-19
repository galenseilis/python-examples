from typing import List
from pydantic import BaseModel

class Message(BaseModel):
    sender: str
    receiver: str
    content: str

messages: List[Message] = []

