import traceback

from textual.app import App, ComposeResult
from textual.widgets import Input, Header, Footer
from textual.containers import VerticalScroll
from textual import work

from langchain.messages import HumanMessage, AIMessage

from TUI.widgets import ChatMessage
from agent.graph import agent

import logging 
logging.basicConfig(filename='travel_agent.log', level=logging.DEBUG)

class TravelAgentApp(App):
    CSS_PATH = "styles.tcss"

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="chat-container")
        yield Input(placeholder="Ask anything...", id="message")
        yield Footer()

    def on_mount(self) -> None:
        self.history = []
        self.agent_state = {}
        self.last_status = None
        self.agent_running = False

        self.add_assistant_message("Hola, ¿a dónde quieres viajar?")

    # =====================================================
    # UI HELPERS
    # =====================================================

    def scroll_to_bottom(self):
        chat = self.query_one("#chat-container")
        chat.scroll_end(animate=False)

    def add_user_message(self, message: str):
        chat = self.query_one("#chat-container")
        self.call_after_refresh(
            lambda: chat.mount(ChatMessage(message, role="user"))
        )
        self.call_after_refresh(self.scroll_to_bottom)

    def add_assistant_message(self, message: str):
        chat = self.query_one("#chat-container")
        self.call_after_refresh(
            lambda: chat.mount(ChatMessage(message, role="assistant"))
        )
        self.call_after_refresh(self.scroll_to_bottom)

    def add_status_message(self, message: str):
        if not message:
            return
        chat = self.query_one("#chat-container")
        self.call_after_refresh(
            lambda: chat.mount(ChatMessage(message, role="status"))
        )
        self.call_after_refresh(self.scroll_to_bottom)

    def set_loading(self, loading: bool):
        self.query_one(Input).disabled = loading

    # =====================================================
    # EVENTS
    # =====================================================

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = self.query_one("#message", Input).value
        self.send_message(message)

    # =====================================================
    # SEND MESSAGE
    # =====================================================

    def send_message(self, message: str) -> None:
        if self.agent_running:
            self.add_status_message("Espera a que termine la búsqueda actual...")
            return

        message = message.strip()
        if not message:
            return

        self.add_user_message(message)
        self.query_one("#message", Input).clear()
        self.run_agent(message)

    # =====================================================
    # AGENT
    # =====================================================

    @work(thread=True)
    def run_agent(self, message: str) -> None:
        self.agent_running = True
        self.call_from_thread(self.set_loading, True)
        self.last_status = None

        NODE_MESSAGES = {
            "flight_node": "Buscando vuelos...",
            "hotel_node": "Buscando hoteles...",
            "activity_node": "Buscando actividades...",
            "budget_node": "Calculando presupuesto...",
            "extract_trip_info": "Analizando tu solicitud...",
            "response_node": "Preparando respuesta...",
        }

        pending_history = [*self.history, HumanMessage(message)]

        input_state = {
            **self.agent_state,
            "messages": pending_history,
        }

        # Acumula el estado de todos los nodos durante el stream
        accumulated_state: dict = {**self.agent_state}
        last_ai_message: str | None = None

        try:
            for chunk in agent.stream(input_state, stream_mode="updates"):
                node_name = list(chunk.keys())[0]
                node_data = chunk[node_name]

                # Mostrar status del nodo actual
                status = NODE_MESSAGES.get(node_name)
                if status and status != self.last_status:
                    self.last_status = status
                    self.call_from_thread(self.add_status_message, status)

                # Merge incremental del estado — nunca sobreescribir con None
                for k, v in node_data.items():
                    if k == "messages":
                        continue
                    if v is not None:
                        accumulated_state[k] = v

                # Capturar el último AIMessage generado
                messages = node_data.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        last_ai_message = last_msg.content

            if last_ai_message is None:
                self.call_from_thread(
                    self.add_assistant_message,
                    "No pude generar una respuesta. Intenta de nuevo."
                )
                return

            # Limpiar tags <think> de modelos como DeepSeek/Qwen
            import re
            response = re.sub(
                r"<think>.*?</think>", "", last_ai_message, flags=re.DOTALL
            ).strip()

            # Persistir estado limpio (sin messages) para el siguiente turno
            self.agent_state = {
                k: v for k, v in accumulated_state.items()
                if k != "messages"
            }

            # Actualizar historial con el turno completo
            self.history = pending_history + [AIMessage(content=response)]

            self.call_from_thread(self.add_assistant_message, response)

        except Exception:
            traceback.print_exc()
            self.call_from_thread(
                self.add_assistant_message,
                "Lo siento, ocurrió un error procesando tu solicitud. Intenta de nuevo."
            )

        finally:
            self.agent_running = False
            self.call_from_thread(self.set_loading, False)

    # =====================================================
    # ACTIONS
    # =====================================================

    def action_toggle_dark(self) -> None:
        self.theme = (
            "textual-dark"
            if self.theme == "textual-light"
            else "textual-light"
        )