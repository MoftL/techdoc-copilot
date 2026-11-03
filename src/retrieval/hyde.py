from core.llm_client import generate_text, get_embedding

def generate_hyde_vector(question: str) -> list[float]:
    system_prompt = "Write a concise, hypothetical technical documentation snippet that directly answers this question."
    user_prompt = f"QUESTION: {question}\nHYPOTHETICAL EXCERPT:"
    hypothetical_doc = generate_text(system_prompt=system_prompt, user_prompt=user_prompt)
    return get_embedding(hypothetical_doc)