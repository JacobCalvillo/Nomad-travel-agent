# TUI/widgets.py

from textual.widgets import Static


class ChatMessage(Static):
    def __init__(self,message: str,role: str = "assistant"):
        safe_message = str(message or "").strip()

        prefix = {
            "user": ">",
            "assistant": "",
            "status": "•"
        }.get(role, "")

        content = f"{prefix} {safe_message}"

        super().__init__(content,classes=f"message {role}",markup=False)