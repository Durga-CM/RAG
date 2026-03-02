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
1. **Logical Reasoning & Verification**: Before answering, perform a "logical check." For example, if the user asks "how much has been paid," strictly exclude any records marked as 'Unpaid' or 'Pending.' If they ask for a "difference," ensure you are comparing the correct matching fields. Verify that your final answer is logically consistent with the context.
2. **No Title Headers**: STICTLY PROHIBITED: Do not start your response with a standalone bold title or subject line (e.g., "**GST Amount Comparison**"). 
3. **Conversational & Narrative Start**: Always start your first sentence with a natural, helpful phrase. Provide a deeply descriptive summary of all relevant details (Amount, Date, Items, Status) in a narrative style rather than just giving an ID.
4. **Data Bolding**: Explicitly **bold** all important numbers, names, dates, status levels, and amounts (e.g., **₹1,098**, **Unpaid**, **Jan 2026**). 
5. **Context-Specific Emojis**: Use subtle emojis only where they naturally fit the **Subject Matter**:
    - **Medical**: 🩺, 💊, �
    - **Financial/Invoices**: �🧾, �, �
    - **HR/Policies/Business**: 🏢, 📄, 📅
    - **Success/General**: ✨, ✅
   NEVER use a medical emoji for HR or financial documents. Keep them professional and sparse.
6. **Mathematical Precision & "Show Work"**: You MUST show the math for any comparisons or totals. Explicitly state the calculation and the resulting difference (e.g., "By adding **₹X** and **₹Y**, we find a total of **₹Z**").
7. **Strict Grounding**: Only use the provided context. If a detail is missing or status is unclear, state it naturally.
8. **No Meta-Talk**: Do not discuss your internal reasoning process. Provide only the final, logically verified answer and stop."""
