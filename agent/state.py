from typing import Annotated, TYPE_CHECKING
from langgraph.graph.message import add_messages

if TYPE_CHECKING:
    from agent.tools import (
        SearchFlightsResponse,
        SearchHotelsResponse,
        GetActivitiesResponse,
    )


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

    # Historial de mensajes (LangGraph lo gestiona con add_messages)
    messages: Annotated[list, add_messages]