from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Optional, Tuple
from typing_extensions import TypedDict
import torch
from llama import Llama
import json
import os

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = List[Message]

class Chat(BaseModel):
    messages: List[Message]

def load_model(ckpt_dir: str, tokenizer_path: str):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=512,
        max_batch_size=4,
        model_parallel_size=1, # The value depends on the number of GPUs you have
    )
    return generator

# Load your model
model = load_model("/home/ubuntu/llama/llama-2-7b-chat/", "tokenizer.model")

app = FastAPI()

def save_to_json_file(data, filename):
    if os.path.exists(filename):
        # If file exists, read the existing data
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    else:
        # If file does not exist, initialize an empty list
        existing_data = []

    # Append new data
    existing_data.append(data)

    # Write back to the file
    with open(filename, 'w') as f:
        json.dump(existing_data, f)

@app.post("/complete-chat/")
async def complete_chat(chat: Chat):
    # Ensure the dialog is in the right format
    dialog = [dict(message) for message in chat.messages]

    # Generate a response
    results = model.chat_completion(
        dialogs=[dialog],
        temperature=0.6,
        top_p=0.9,
    )

    # Get the assistant's message from results
    assistant_msg = results[0]['generation']['content']

    # Save the chat input and output to a JSON file
    save_to_json_file({
        "input": dialog,
        "output": assistant_msg
    }, "chat_data.json")

    return {"assistant": assistant_msg}
