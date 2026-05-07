from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class Context(TypedDict, total=False):
    persons: int
    children: int
    budget: float

    place: str
    origin: str

    arrival_date: str
    leave_date: str

    flights: list | None
    hotels: list | None
    activities: list | None

    itinerary: str | None

    wants_flights: bool
    wants_hotels: bool
    wants_activities: bool

    budget_total: float | None

    messages: Annotated[list, add_messages]

    llm_calls: int