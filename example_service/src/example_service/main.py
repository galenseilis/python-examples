from fastapi import FastAPI, HTTPException
from typing import List
from models import Message, messages
from schemas import MessageCreate

app = FastAPI()

@app.post("/send_message/", response_model=Message)
def send_message(message: MessageCreate):
    new_message = Message(sender=message.sender, receiver=message.receiver, content=message.content)
    messages.append(new_message)
    return new_message

@app.get("/get_messages/{receiver}", response_model=List[Message])
def get_messages(receiver: str):
    receiver_messages = [message for message in messages if message.receiver == receiver]
    if not receiver_messages:
        raise HTTPException(status_code=404, detail="No messages found for this receiver")
    return receiver_messages

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
