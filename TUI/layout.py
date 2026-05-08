import traceback
import re
import logging

from textual.app import App, ComposeResult
from textual.widgets import Input, Header, Footer
from textual.containers import VerticalScroll
from textual import work

from langchain.messages import HumanMessage, AIMessage

from TUI.widgets import ChatMessage
from agent.graph import agent

logging.basicConfig(filename='travel_agent.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
        container = self.query_one("#chat-container")
        container.call_after_refresh(container.scroll_end, animate=True)

    def add_message(self, message: str, role: str = "assistant"):
        """Método genérico para añadir mensajes al chat."""
        if not message:
            return
        chat = self.query_one("#chat-container")
        self.call_after_refresh(
            lambda: chat.mount(ChatMessage(message, role=role))
        )
        self.call_after_refresh(self.scroll_to_bottom)

    def add_user_message(self, message: str):
        """Añade un mensaje del usuario."""
        self.add_message(message, role="user")

    def add_assistant_message(self, message: str):
        """Añade un mensaje del asistente."""
        self.add_message(message, role="assistant")

    def add_status_message(self, message: str):
        """Añade un mensaje de estado."""
        self.add_message(message, role="status")

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

        # 1. Preparamos el historial actual incluyendo el mensaje del usuario
        current_human_msg = HumanMessage(content=message)
        pending_history = [*self.history, current_human_msg]

        input_state = {
            **self.agent_state,
            "needs_clarification": False,
            "clarification_messages": [],
            "messages": pending_history,
        }

        # Acumuladores
        accumulated_state: dict = {**self.agent_state}
        last_ai_message: str | None = None

        try:
            # 2. Streaming del agente
            for chunk in agent.stream(input_state, stream_mode="updates"):
                # IMPORTANTE: Validar que el chunk sea un diccionario
                if not isinstance(chunk, dict):
                    continue

                for node_name, node_data in chunk.items():
                    # FIX: Validar que node_data no sea None
                    if node_data is None:
                        logger.debug(f"Nodo {node_name} envió data vacía (None)")
                        continue

                    # Actualizar estado visual (Status)
                    status = NODE_MESSAGES.get(node_name)
                    if status and status != self.last_status:
                        self.last_status = status
                        self.call_from_thread(self.add_status_message, status)

                    # 3. Mezcla incremental de datos
                    # Usamos .get() por seguridad
                    for k, v in node_data.items():
                        if k == "messages" or v is None:
                            continue
                        accumulated_state[k] = v

                    # 4. Captura el mensaje de respuesta
                    messages = node_data.get("messages", [])
                    if messages:
                        ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
                        if ai_msgs:
                            content = ai_msgs[-1].content
                            # SEGURIDAD: Si el contenido es una lista (multimodal/tools), 
                            # extraer solo el texto
                            if isinstance(content, list):
                                text_parts = [
                                    part.get("text", "") if isinstance(part, dict) else str(part)
                                    for part in content
                                ]
                                last_ai_message = "".join(text_parts)
                            else:
                                last_ai_message = content
            # 5. Validación de respuesta
            if last_ai_message is None:
                return 

            # Limpiar tags de razonamiento (DeepSeek/Qwen)
            response = re.sub(r"<think>.*?</think>", "", str(last_ai_message), flags=re.DOTALL).strip()

            # 6. PERSISTENCIA CRÍTICA:
            # Actualizamos el estado global de la App para el siguiente turno
            self.agent_state = {
                k: v for k, v in accumulated_state.items()
                if k != "messages"
            }

            # Actualizamos el historial real de la conversación
            self.history.append(current_human_msg)
            self.history.append(AIMessage(content=response))

            # Mostramos la respuesta en la UI
            self.call_from_thread(self.add_assistant_message, response)

        except Exception as e:
            # Loguear el error real para debug
            logger.error(f"Error en run_agent: {traceback.format_exc()}")
            self.call_from_thread(
                self.add_assistant_message,
                f"Lo siento, ocurrió un error: {str(e)}" 
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