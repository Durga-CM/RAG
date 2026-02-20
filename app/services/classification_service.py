import ollama
from app.core.config import CLASSIFICATION_MODEL

class ClassificationService:

    def __init__(self):
        self._cache = {}

    def classify_document(self, text: str) -> str:
        """
        🔥 PRODUCTION-GRADE: Detect document type by analyzing actual content using LLM.
        Directly from invoice_rag_standalone.py
        """
        if not text or len(text.strip()) < 50:
            return "general"
        
        sample = text[:2000]
        
        classification_prompt = f"""You are an expert document classifier for a production RAG system.

Analyze the following document text and classify it into ONE category:

**Categories:**
- medical: Health/medical records — lab reports, patient data, prescriptions, diagnoses, hospital records, doctor's notes
- invoice: Financial/billing documents — bills, invoices, receipts, tax documents, financial statements, payment records
- hr: Employee/workforce documents — employee records, payroll, recruitment, performance reviews, salary, attendance, leave policies
- insurance_policy: Insurance documents — vehicle/motor/health/life/property insurance policies, policy schedules, premium breakdowns, IDV, coverage details, insurer details
- general: Any document that doesn't clearly fit the above categories

**Document Text:**
{sample}

**Critical Decision Rules:**
1. Insurance vs Invoice:
   - If document mentions: Policy Number, Insurer, IDV, Premium Payable, Coverage, Insured Name, Registration Number of vehicle → INSURANCE_POLICY
   - Bills, invoices, payment receipts for goods/services → INVOICE
2. Look for these specific indicators:
   - Medical: "Patient Name", "Diagnosis", "Doctor", "Hospital", "Lab Results", "Blood Test"
   - Invoice: "Amount Payable", "GST", "Bill Number", "Tax Invoice", "Supplier", "Recipient"
   - HR: "Employee ID", "Salary", "Department", "Designation", "Leave", "Payroll"
   - Insurance: "Policy Number", "Insurer", "IDV", "Premium", "Policy Period", "Insured", "Vehicle Registration"
3. Return ONLY one word: medical, invoice, hr, insurance_policy, or general
4. No explanation, no punctuation, just the category label

**Category:**"""

        try:
            response = ollama.chat(
                model=CLASSIFICATION_MODEL,
                messages=[{"role": "user", "content": classification_prompt}]
            )
            doc_type = response["message"]["content"].strip().lower()
            valid_types = ["medical", "invoice", "hr", "insurance_policy", "general"]
            if doc_type in valid_types:
                return doc_type
            for vtype in valid_types:
                if vtype in doc_type:
                    return vtype
            return "general"
        except Exception:
            return "general"

    def classify_query(self, query: str) -> str | None:
        """
        🔥 PRODUCTION-GRADE: LLM-based query classification.
        Directly from invoice_rag_standalone.py
        """
        query_key = query.lower().strip()
        if query_key in self._cache:
            return self._cache[query_key]
        
        classification_prompt = f"""You are a document type classifier.

Classify the following user query into ONE of these categories:
- medical (health records, lab reports, patient data, prescriptions, diagnosis, doctor, hospital)
- invoice (bills, payments, tax documents, goods/services billing, GST, supplier, recipient)
- hr (employee data, payroll, recruitment, workforce, leave policy, salary, attendance)
- insurance_policy (insurance policies, premium, IDV, policy number, insurer, coverage, vehicle insurance, policy period, insured name)
- general (if the query doesn't clearly fit any specific category)

USER QUERY: "{query}"

RULES:
1. Return ONLY the category label (one word: medical, invoice, hr, insurance_policy, or general)
2. No explanation, no punctuation, just the label
3. If unsure, return "general"
4. CRITICAL: Queries about "premium", "policy number", "IDV", "insurer", "insured", "policy period", "coverage" → ALWAYS insurance_policy
5. Queries about "GST", "bill", "supplier", "items bought" → invoice
6. HR is ONLY for non-financial topics like Leave, Attendance, Hiring, or Roles.

CATEGORY:"""

        try:
            response = ollama.chat(
                model=CLASSIFICATION_MODEL,
                messages=[{"role": "user", "content": classification_prompt}]
            )
            doc_type = response["message"]["content"].strip().lower()
            valid_types = ["medical", "invoice", "hr", "insurance_policy", "general"]

            result = None
            if doc_type in valid_types:
                result = doc_type if doc_type != "general" else None
            else:
                for vtype in valid_types:
                    if vtype in doc_type:
                        result = vtype if vtype != "general" else None
                        break

            self._cache[query_key] = result
            return result
        except Exception:
            query_lower = query.lower()
            if any(word in query_lower for word in ["insurance", "premium", "idv", "policy number", "insurer", "insured", "policy period", "coverage", "vehicle registration"]):
                result = "insurance_policy"
            elif any(word in query_lower for word in ["medical", "patient", "doctor", "hospital", "diagnosis", "wbc", "blood", "health", "hemoglobin", "lab", "prescription"]):
                result = "medical"
            elif any(word in query_lower for word in ["invoice", "bill", "payment", "amount", "gst", "tax", "supplier", "items"]):
                result = "invoice"
            elif any(word in query_lower for word in ["hr", "employee", "payroll", "recruitment", "salary", "leave", "workforce"]):
                result = "hr"
            else:
                result = None
            self._cache[query_key] = result
            return result

    def detect_filename_in_query(self, query: str) -> str | None:
        """
        Detect if user is asking about a specific file by name.
        Directly from invoice_rag_standalone.py
        """
        query_lower = query.lower()
        patterns = [
            "medical_report",
            "medical report",
            "hr_monthly_report",
            "hr monthly report",
            "invoice_",
            "invoice ",
            "insurance_policy",
            "insurance policy",
            "insurance ",
        ]
        for pattern in patterns:
            if pattern in query_lower:
                return pattern.replace(" ", "_").strip("_")
        return None
