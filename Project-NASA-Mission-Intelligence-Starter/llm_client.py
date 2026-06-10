from typing import Dict, List
from openai import OpenAI


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo"
) -> str:
    """Generate response using OpenAI with retrieved NASA context"""

    system_prompt = """
You are a NASA mission intelligence assistant.

Answer questions using ONLY the retrieved NASA mission context provided.
If the context does not contain enough information, clearly say that the available context is insufficient.
Do not make unsupported claims.
Cite the retrieved source names when possible.
Keep answers clear, factual, and useful for mission analysis.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""
Retrieved Context:
{context}

User Question:
{user_message}
"""
        }
    ]

    # Add limited conversation history
    if conversation_history:
        recent_history = conversation_history[-6:]
        messages = [{"role": "system", "content": system_prompt}] + recent_history + [
            {
                "role": "user",
                "content": f"""
Retrieved Context:
{context}

User Question:
{user_message}
"""
            }
        ]

    try:
        client = OpenAI(api_key=openai_key)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating response: {e}"
