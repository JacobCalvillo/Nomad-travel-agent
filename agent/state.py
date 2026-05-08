from typing import Annotated
import operator
from langgraph.graph.message import add_messages

from typing import TypedDict


class Context(TypedDict, total=False):
    # Viajeros
    persons: int
    children: int

    # Presupuesto indicado por el usuario
    budget: float

    # Destino
    place: str
    origin: str

    # Fechas (ISO 8601: YYYY-MM-DD)
    arrival_date: str
    leave_date: str

    # Resultados de herramientas
    flights: list | None
    hotels: list | None
    activities: list | None

    # Itinerario opcional (para uso futuro)
    itinerary: str | None

    # Flags de intención extraídos del mensaje del usuario
    wants_flights: bool
    wants_hotels: bool
    wants_activities: bool

    # Presupuesto calculado
    budget_total: float | None
    
    needs_clarification: Annotated[
        bool,
        lambda a, b: a or b
    ]

    clarification_messages: Annotated[
        list[str],
        operator.add
    ]

    # Historial de mensajes (LangGraph lo gestiona con add_messages)
    messages: Annotated[list, add_messages]