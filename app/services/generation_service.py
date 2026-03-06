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
            # Include the filename so the AI can refer to it by name
            fname = ctx.get('file_name', 'Unknown Document')
            context_block += f"\n--- DOCUMENT: {fname} ---\n{ctx['content']}\n"

        history_block = f"\nRECENT CHAT HISTORY:\n{history_text}\n" if history_text else ""

        return f"""You are a high-precision document analysis assistant. Your goal is to provide logically sound, accurate, and conversational answers. You must use critical thinking to interpret the user's intent and verify that your answer makes logical sense based on all provided details (like status, dates, and amounts).

CONTEXT DATA:
{context_block}
{history_block}

GUIDELINES FOR YOUR RESPONSE:
1. **Direct Start (STRICT)**: STIRCTLY PROHIBITED: Do not use any introductory phrases (e.g., "Let's analyze," "Sure," "Based on the context," "Hmm," "According to"), thinking emojis, or filler sentences. Your response MUST start immediately with the factual data or a direct answer.
2. **Natural Emojis (Context-Aware)**: Use subtle, relevant emojis to make the response engaging. 
    - **Invoices**: 🧾, 💰
    - **Medical**: 🩺, 💊
    - **Policies/Corporate**: 🏢, 📄
    - **Success**: ✨
    CRITICAL: Never use "thinking" emojis (🤔) or "questioning" emojis (❓). Only use icons that represent the document type you found. If you find multiple types, use ✨.
3. **Focus on Positives**: Do not apologize or mention what was NOT found unless everything is missing. If you found "Parthiban" in an insurance document, do not mention that he wasn't in the medical records.
4. **Narrative Grouping (NO TABLES)**: STIRCTLY PROHIBITED: Do not use Markdown Tables. Group related facts into clean, descriptive paragraphs. If there are multiple items, describe them naturally within the narrative rather than using tables or complex lists.
5. **Data Bolding**: Only **bold** truly critical data points (names, amounts, status, dates).
6. **Smart Entity Matching**: Treat "Deepak Raj" and "DEEPAK RAJ N" as the same person.
7. **Mathematical Precision**: Show your work for any comparisons or additions.
8. **Avoid Repetition**: Mention header metadata (like State or Year) only once per response. Group all specifics under a unified summary.
9. **Strict Grounding**: Use ONLY the provided context. If data is genuinely missing from ALL provided files, state it naturally without apology.
10. **Logical Integrity (Sanity Check)**: Before stating a conclusion, verify it is mathematically sound. Never state a logical contradiction (e.g., do not say "2 items is more than 2 items"). If counts or values are equal, strictly state they are equal. """
