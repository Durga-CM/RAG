import ollama
from app.core.config import LLM_MODEL

class GenerationService:

    def generate(self, query, contexts, extracted_info=None):
        """
        Final answer generation.
        Uses a strict Hierarchy of Truth to prevent mixing values.
        """
        # Formulate context block
        context_block = ""
        for idx, ctx in enumerate(contexts):
            context_block += f"\n--- DOCUMENT {idx+1} Content ({ctx['file_name']}) ---\n{ctx['content']}\n"

        system_prompt = f"""You are a professional document analysis agent.
Your task is to answer the user's question using ONLY the [DOCUMENT Content] provided.

CONTENT:
{context_block}

STRICT OPERATIONAL RULES:
1. ONLY use data labeled in the document body (e.g. 'Invoice No:', 'Reference No:').
2. DO NOT use the filename metadata (e.g. 'Invoice_2500194869') as a data value.
3. If an extraction hint is provided ('{extracted_info}'), use it only to locate the relevant field in the documents — do NOT treat it as the sole answer if multiple documents each have their own distinct value.
4. If multiple different values are found across documents, list ALL of them and clearly state which document each comes from. Do not pick a "primary" one arbitrarily.
5. Provide a direct, factual answer. Do not apologize or explain your steps."""

        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"Error: {str(e)}"
