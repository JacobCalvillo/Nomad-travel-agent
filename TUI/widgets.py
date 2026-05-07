from textual.widgets import Static, Markdown
from textual.app import ComposeResult


class ChatMessage(Static):
    """
    Mensaje de chat genérico.
    - role='assistant' → renderiza Markdown
    - role='user'      → texto plano con prefijo '>'
    - role='status'    → texto plano con prefijo '•'
    """

    DEFAULT_CSS = """
    ChatMessage {
        width: 1fr;
        padding: 0 1;
    }
    ChatMessage.user {
        color: $accent;
    }
    ChatMessage.status {
        color: $text-muted;
        opacity: 0.7;
    }
    ChatMessage.assistant {
        padding: 0;
    }
    """

    def __init__(self, message: str, role: str = "assistant"):
        self._message = str(message or "").strip()
        self._role = role
        super().__init__(classes=f"message {role}")

    def compose(self) -> ComposeResult:
        if self._role == "assistant":
            yield Markdown(self._message)
        elif self._role == "user":
            yield Static(f"> {self._message}", markup=False, classes="user-text")
        else:
            yield Static(f"• {self._message}", markup=False, classes="status-text")