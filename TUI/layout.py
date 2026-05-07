from textual.app import App, ComposeResult
from textual.widgets import RichLog, Input, Header, Footer, Button
from textual.containers import Horizontal
from textual import work

from langchain.messages import HumanMessage

from agent.graph import agent



class TravelAgentApp(App):
    
    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(markup=True)
        with Horizontal():
            yield Input(placeholder="Escribe tu mensaje...", id='message')
            yield Button("Enviar")
        yield Footer()
    
    def on_mount(self) -> None:
        self.history = []
        self.agent_state = {}
        self.last_status = None
        self.agent_running = False
        
        
        log = self.query_one(RichLog)
        log.write("[orange1] Hola, ¿a dónde quieres viajar?[/orange1]")
        
        
    def on_button_pressed(self, event: Button.Pressed):
        message = self.query_one('#message', Input).value
        self.send_message(message)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = self.query_one('#message', Input).value
        self.send_message(message)
        
    def send_message(self, message: str) -> None:
        if self.agent_running:
            return
        
        message = message.strip()
        
        if not message:
            return
        
        log = self.query_one(RichLog)
        log.write(f"Tú: {message}")
        self.query_one('#message', Input).clear()
        self.run_agent(message)
    
    @work(thread=True)
    def run_agent(self, message: str) -> None:
        self.agent_running = True
        self.last_status = None

        TOOL_MESSAGES = {
            "search_flights": "Buscando vuelos...",
            "search_hotels": "Buscando hoteles...",
            "get_activities": "Buscando actividades...",
            "calc_budget": "Calculando presupuesto...",
        }

        pending_history = [*self.history, HumanMessage(message)]

        input_state = {
            **self.agent_state,
            "messages": pending_history,
            "llm_calls": 0
        }

        final_state = None

        try:
            for chunk in agent.stream(input_state,stream_mode="values"):
                final_state = chunk
                messages = chunk.get("messages", [])

                if not messages:
                    continue
                
                last_message = messages[-1]
                if getattr(last_message, "tool_calls", None):
                    for tc in last_message.tool_calls:
                        status = TOOL_MESSAGES.get(
                            tc["name"],
                            f"⚙ {tc['name']}..."
                        )
                        if status != self.last_status:
                            self.last_status = status
                            self.call_from_thread(self._write_status, status)


            if not final_state:
                return

            self.agent_state = {
                k: v
                for k, v in final_state.items()
                if k != "messages"
            }

            self.history = final_state["messages"]
            respuesta = (final_state["messages"][-1].content)

            self.call_from_thread(self._write_response,respuesta)

        except Exception as e:
            self.call_from_thread(
                self._write_response,
                f"[red]Error:[/red] {str(e)}"
            )
        finally:
            self.agent_running = False
        
    def _write_response(self, respuesta:str) -> None:
        log = self.query_one(RichLog)
        log.write(f"[orange1] {respuesta}[/orange1]")

    def _write_status(self, status: str) -> None:
        log = self.query_one(RichLog)
        log.write(f"[dim italic]{status}[/dim italic]")

    
    def action_toggle_dark(self) -> None:
        self.theme = ('textual-dark' if self.theme == 'textual-light' else 'textual-light')