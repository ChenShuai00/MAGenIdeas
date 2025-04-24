"""Module for interacting with Deepseek chat API."""

from typing import Tuple
from openai import OpenAI

def deepseek_chat(client: OpenAI, system_prompt: str, user_prompt: str) -> Tuple[str, int]:
    """Send chat completion request to Deepseek API.
    
    Args:
        client: OpenAI client instance configured for Deepseek
        system_prompt: System message setting chat behavior
        user_prompt: User message with actual prompt content
        
    Returns:
        tuple: (response_text, token_count)
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )
    
    response_text = response.choices[0].message.content
    token_count = response.usage.total_tokens
    
    return response_text, token_count
