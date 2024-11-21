from pydantic import BaseModel


class MessageCreate(BaseModel):
    sender: str
    receiver: str
    content: str
