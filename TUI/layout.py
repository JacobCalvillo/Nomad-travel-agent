
from textual.app import App, ComposeResult
from textual.widgets import Input, Header, Footer
from textual.containers import VerticalScroll
from textual import work

from langchain.messages import HumanMessage

from TUI.widgets import ChatMessage
from agent.graph import agent


class TravelAgentApp(App):
    CSS_PATH = "styles.tcss"

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="chat-container")
        yield Input(placeholder="Ask anything...",id="message")
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

    def add_user_message(self,message: str):
        chat = self.query_one("#chat-container")

        self.call_after_refresh(
            lambda: chat.mount(
                ChatMessage(
                    message,
                    role="user"
                )
            )
        )
        self.call_after_refresh(self.scroll_to_bottom)

    def add_assistant_message(self,message: str):
        chat = self.query_one("#chat-container")

        self.call_after_refresh(lambda: chat.mount(ChatMessage(message,role="assistant")))
        self.call_after_refresh(self.scroll_to_bottom)

    def add_status_message(self,message: str):
        chat = self.query_one("#chat-container")

        self.call_after_refresh(
            lambda: chat.mount(
                ChatMessage(message,role="status")))
        
        self.call_after_refresh(self.scroll_to_bottom)

    def set_loading(self,loading: bool):
        self.query_one(Input).disabled = loading

    # =====================================================
    # EVENTS
    # =====================================================

    def on_input_submitted(self,event: Input.Submitted) -> None:
        message = self.query_one("#message",Input).value
        self.send_message(message)

    # =====================================================
    # SEND MESSAGE
    # =====================================================

    def send_message(self,message: str) -> None:
        if self.agent_running:
            self.add_status_message("Espera a que termine la búsqueda actual...")
            return

        message = message.strip()

        if not message:
            return

        self.add_user_message(message)
        self.query_one("#message",Input).clear()
        self.run_agent(message)

    # =====================================================
    # AGENT
    # =====================================================

    @work(thread=True)
    def run_agent(self,message: str) -> None:

        self.agent_running = True
        self.call_from_thread(self.set_loading,True)
        self.last_status = None

        TOOL_MESSAGES = {
            "search_flights": "Buscando vuelos...",
            "search_hotels": "Buscando hoteles...",
            "get_activities": "Buscando actividades...",
            "calc_budget": "Calculando presupuesto..."
        }

        pending_history = [*self.history,HumanMessage(message)]

        input_state = {
            **self.agent_state,
            "messages": pending_history,
            "llm_calls": 0
        }

        final_state = None

        try:
            for chunk in agent.stream(input_state,stream_mode="values"):
                final_state = chunk
                messages = chunk.get("messages",[])

                if not messages:
                    continue

                last_message = messages[-1]

                if getattr(last_message,"tool_calls",None):
                    for tc in last_message.tool_calls:
                        status = TOOL_MESSAGES.get(tc["name"],f"Running {tc['name']}...")

                        if status != self.last_status:
                            self.last_status = status
                            self.call_from_thread(self.add_status_message,status)

            if not final_state:
                return

            self.agent_state = {
                k: v
                for k, v in final_state.items()
                if k != "messages"
            }

            self.history = final_state["messages"]
            response = final_state["messages"][-1].content

            if "<think>" in response:
                import re

                response = re.sub(
                    r"<think>.*?</think>",
                    "",
                    response,
                    flags=re.DOTALL
                ).strip()

            self.call_from_thread(self.add_assistant_message,response)

        except Exception as e:
            self.call_from_thread(self.add_assistant_message,f"Error: {str(e)}")

        finally:
            self.agent_running = False
            self.call_from_thread(self.set_loading,False)

    # =====================================================
    # ACTIONS
    # =====================================================

    def action_toggle_dark(self) -> None:
        self.theme = (
            "textual-dark"
            if self.theme == "textual-light"
            else "textual-light"
        )