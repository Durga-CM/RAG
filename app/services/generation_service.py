import ollama
from app.core.config import LLM_MODEL

class GenerationService:

    def generate(self, query, contexts, history_text=None):
        """
        Consolidated extraction and final answer generation (Non-streaming).
        """
        system_prompt = self._build_prompt(contexts, history_text)

        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                options={"temperature": 0}
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"Error during generation: {str(e)}"

    def generate_stream(self, query, contexts, history_text=None):
        """
        Streamed LLM generation. Yields token chunks.
        """
        system_prompt = self._build_prompt(contexts, history_text)

        try:
            stream = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                options={"temperature": 0},
                stream=True
            )
            for chunk in stream:
                yield chunk['message']['content']
        except Exception as e:
            yield f"\n[STREAM_ERROR]: {str(e)}"

    def _build_prompt(self, contexts, history_text):
        # Formulate context block
        context_block = ""
        for idx, ctx in enumerate(contexts):
            label = ctx['doc_type'].replace('_', ' ').upper()
            context_block += f"\n[[INTERNAL_ID_{idx+1}: {label}]]\n{ctx['content']}\n"

        history_block = f"\nRECENT CHAT HISTORY:\n{history_text}\n" if history_text else ""

        return f"""You are a helpful and professional document analysis assistant. 

CONTEXT DATA:
{context_block}
{history_block}

GUIDELINES FOR YOUR RESPONSE:
1. **Direct Answer**: Provide a polite, direct answer based ONLY on the context above.
2. **Identification**: Refer to files by their natural names (e.g. "The 2026 Leave Policy", "the ICICI insurance policy").
3. **STRICT PROHIBITION**: 
   - **DO NOT** use generic labels like "Document 1", "Reference 1", or "ID 1".
   - **DO NOT** include internal tags like "(Document 1)" or "[INTERNAL_ID_1]" in your sentences.
   - Speak naturally. If you mention a document, just use its title.
4. **Accuracy**: Do not hallucinate. Do not mix up item prices with subtotals."""
