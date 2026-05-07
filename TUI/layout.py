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
        
        log = self.query_one(RichLog)
        log.write("[orange1] Hola, ¿a dónde quieres viajar?[/orange1]")
        
        
    def on_button_pressed(self, event: Button.Pressed):
        mensaje = self.query_one('#message', Input).value
        self.send_message(mensaje)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        mensaje = self.query_one('#message', Input).value
        self.send_message(mensaje)
        
    def send_message(self, mensaje: str) -> None:
        log = self.query_one(RichLog)
        log.write(f"Tú: {mensaje}")
        self.query_one('#message', Input).clear()
        self.run_agent(mensaje)
    
    @work(thread=True)
    def run_agent(self, mensaje: str) -> None:
        TOOL_MESSAGES = {
            "search_flights": "Buscando vuelos...",
            "search_hotels": "Buscando hoteles...",
            "get_activities": "Buscando actividades...",
            "calc_budget": "Calculando presupuesto...",
        }

        self.history.append(HumanMessage(mensaje))
        input_state = {**self.agent_state, 'messages': self.history}

        result = None
        for chunk in agent.stream(input_state, stream_mode='updates'):
            result = chunk

            if 'llm_call' in chunk:
                msgs = chunk['llm_call'].get('messages', [])
                for msg in msgs:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            status = TOOL_MESSAGES.get(tc['name'], f"⚙ {tc['name']}...")
                            self.call_from_thread(self._write_status, status)

        # obtener el estado final
        final = agent.invoke(input_state)
        self.agent_state = {k: v for k, v in final.items() if k != 'messages'}
        self.history = final['messages']
        respuesta = final['messages'][-1].content
        self.call_from_thread(self._write_response, respuesta)    
        
    def _write_response(self, respuesta:str) -> None:
        log = self.query_one(RichLog)
        log.write(f"[orange1] {respuesta}[/orange1]")

    def _write_status(self, status: str) -> None:
        log = self.query_one(RichLog)
        log.write(f"[dim italic]{status}[/dim italic]")

    
    def action_toggle_dark(self) -> None:
        self.theme = ('textual-dark' if self.theme == 'textual-light' else 'textual-light')