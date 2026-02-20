import ollama
import re
from app.core.config import LLM_MODEL

class ExtractionService:

    def extract(self, text, query):
        """
        Robotic data extraction. 
        Ensures ONLY the literal value is returned to prevent confusing the generator.
        """
        extraction_prompt = f"""[SYSTEM: ROBOTIC DATA EXTRACTOR]
TASK: Extract the literal value for '{query}' from the text.
RULES:
1. Return ONLY the value. 
2. NO sentences, NO filler, NO "The value is".
3. If not found, return 'N/A'.

TEXT:
{text[:12000]}

VALUE:"""
        try:
            # Use generate with a strict prompt
            response = ollama.generate(
                model=LLM_MODEL,
                prompt=extraction_prompt,
                options={"temperature": 0, "num_predict": 300}
            )
            result = response["response"].strip()
            
            # --- AGGRESSIVE CLEANUP ---
            # Remove any "Sure!" or "I've got it" or "Value:" labels
            result = re.sub(r'(?i)^(Value|Answer|Result|The value is|I found|Sure|Here is|Literal Value)[:\s]*', '', result)
            
            # If the model returned multiple lines, take the most likely one (first or one with digits)
            if "\n" in result:
                lines = [line.strip() for line in result.split("\n") if line.strip()]
                for line in lines:
                    if any(c.isdigit() for c in line) and len(line) < 40:
                        result = line
                        break
                else:
                    result = lines[0]

            # Final check: if it looks like a sentence, it's likely a hallucination/filler
            if len(result) > 300 or "cannot find" in result.lower():
                return None
                
            return result if result.lower() != "n/a" else None
        except Exception:
            return None
