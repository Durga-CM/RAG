from sqlalchemy.orm import Session
from app.models.models import ChatHistory

class HistoryService:
    @staticmethod
    def save_message(db: Session, session_id: str, role: str, content: str):
        message = ChatHistory(session_id=session_id, role=role, content=content)
        db.add(message)
        db.commit()
        db.refresh(message)
        return message

    @staticmethod
    def get_history(db: Session, session_id: str, limit: int = 10):
        return db.query(ChatHistory).filter(ChatHistory.session_id == session_id).order_by(ChatHistory.timestamp.asc()).limit(limit).all()

    @staticmethod
    def clear_history(db: Session, session_id: str):
        db.query(ChatHistory).filter(ChatHistory.session_id == session_id).delete()
        db.commit()

    @staticmethod
    def format_history_for_llm(history):
        # Already ascending, so no need to reverse if it's already in chronological order
        # Wait, get_history was descending. I'll change it to ascending for easier processing.
        formatted = []
        for msg in history:
            role_label = "User" if msg.role == "user" else "Assistant"
            formatted.append(f"{role_label}: {msg.content}")
        return "\n".join(formatted)
