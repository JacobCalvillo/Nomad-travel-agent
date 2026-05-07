import os
import logging
from datetime import date

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain.messages import SystemMessage, AIMessage, HumanMessage

from agent.tools import search_flights, search_hotels, get_activities, calc_budget
from agent.state import Context

logger = logging.getLogger(__name__)


# =========================================================
# Structured extraction schema
# =========================================================

class TripInfo(BaseModel):
    persons: int = 1
    children: int = 0

    budget: float | None = None

    place: str | None = None
    origin: str | None = None

    arrival_date: str | None = None
    leave_date: str | None = None

    wants_flights: bool = False
    wants_hotels: bool = False
    wants_activities: bool = False


# =========================================================
# Models
# =========================================================

model = init_chat_model(model=f"google_genai:{os.getenv('GEMINI_MODEL')}")

extractor = model.with_structured_output(TripInfo)


# =========================================================
# Extractor Node
# =========================================================

def extract_trip_info(state: Context) -> dict:
    # Construir contexto con las últimas 3 rondas de conversación
    recent_messages = state.get("messages", [])[-6:]
    conversation_context = "\n".join(
        f"{'Usuario' if isinstance(m, HumanMessage) else 'Asistente'}: {m.content}"
        for m in recent_messages
        if hasattr(m, "content") and m.content
    )

    try:
        extracted: TripInfo = extractor.invoke(
            f"""
            Extract travel information from the conversation below.
            If the user refers to something already established (e.g., "solo yo", "el mismo día"),
            use the current trip state to fill in the values — don't leave them null.

            Rules:
            - Return numbers as numbers
            - Dates must use YYYY-MM-DD format
            - "solo yo" or "para mí solo" means persons=1
            - "el mismo dia" means use the same arrival_date from current state

            Today is {date.today()}.

            Current trip state:
            Origin: {state.get("origin")}
            Destination: {state.get("place")}
            Arrival date: {state.get("arrival_date")}
            Leave date: {state.get("leave_date")}
            Budget: {state.get("budget")}
            Persons: {state.get("persons")}
            Children: {state.get("children")}

            Conversation:
            {conversation_context}
            """
        )
    except Exception:
        # Si la extracción falla, preservar el estado actual sin cambios
        extracted = TripInfo(
            persons=state.get("persons", 1),
            children=state.get("children", 0),
            budget=state.get("budget"),
            place=state.get("place"),
            origin=state.get("origin"),
            arrival_date=state.get("arrival_date"),
            leave_date=state.get("leave_date"),
            wants_flights=False,
            wants_hotels=False,
            wants_activities=False,
        )

    # ------------------------------------
    # Fix fechas invertidas
    # ------------------------------------
    arrival = extracted.arrival_date
    leave = extracted.leave_date

    if arrival and leave:
        arrival_dt = date.fromisoformat(arrival)
        leave_dt = date.fromisoformat(leave)
        if leave_dt < arrival_dt:
            arrival, leave = leave, arrival

    # ------------------------------------
    # Construir updates — solo campos con valor
    # ------------------------------------
    updates: dict = {}

    if extracted.persons is not None:
        updates["persons"] = extracted.persons
    if extracted.children is not None:
        updates["children"] = extracted.children
    if extracted.budget is not None:
        updates["budget"] = extracted.budget
    if extracted.place:
        updates["place"] = extracted.place
    if extracted.origin:
        updates["origin"] = extracted.origin
    if arrival:
        updates["arrival_date"] = arrival
    if leave:
        updates["leave_date"] = leave

    updates["wants_flights"] = extracted.wants_flights
    updates["wants_hotels"] = extracted.wants_hotels
    updates["wants_activities"] = extracted.wants_activities

    logger.debug(f"EXTRACTED: {updates}")
    return updates


# =========================================================
# Tool Nodes
# =========================================================

def flight_node(state: Context) -> dict:
    if not state.get("arrival_date"):
        return {
            "messages": [AIMessage(content="¿Qué fecha de salida tienes en mente para el vuelo?")]
        }

    # Validar que la fecha no sea pasada
    if date.fromisoformat(state["arrival_date"]) < date.today():
        return {
            "messages": [AIMessage(content="La fecha de salida ya pasó. ¿Cuándo quieres viajar?")]
        }

    if not state.get("origin"):
        return {
            "messages": [AIMessage(content="¿Desde qué ciudad vas a salir?")]
        }

    if not state.get("place"):
        return {
            "messages": [AIMessage(content="¿A qué destino quieres viajar?")]
        }

    flights = search_flights.invoke({
        "passengers": state.get("persons", 1),
        "origin": state["origin"],
        "destination": state["place"],
        "arrival_date": state["arrival_date"],
        "leave_date": state.get("leave_date"),
        "type_of_flight": "Redondo" if state.get("leave_date") else "Sencillo",
    })
    
    if not flights:
        return {
            "flights": [],
            "message": AIMessage(content="No encontré vuelos directos para esa ruta y fecha. Prueba con otras fechas o aerolíneas.")
        }

    return {"flights": flights}


def hotel_node(state: Context) -> dict:
    if not state.get("arrival_date") or not state.get("leave_date"):
        return {
            "messages": [AIMessage(content="¿Qué fechas necesitas para el hospedaje? (entrada y salida)")]
        }

    if not state.get("place"):
        return {
            "messages": [AIMessage(content="¿En qué ciudad buscamos hotel?")]
        }

    hotels = search_hotels.invoke({
        "place": state["place"],
        "check_in_date": state["arrival_date"],
        "check_out_date": state["leave_date"],
        "adults": state.get("persons", 1),
        "children": state.get("children", 0),
    })

    return {"hotels": hotels}


def activity_node(state: Context) -> dict:
    if not state.get("place"):
        return {
            "messages": [AIMessage(content="¿En qué ciudad buscamos actividades?")]
        }

    activities = get_activities.invoke({"place": state["place"]})
    return {"activities": activities}


def budget_node(state: Context) -> dict:
    flights = state.get("flights") or []
    hotels = state.get("hotels") or []
    activities = state.get("activities") or []

    cheapest_flight = min((f.price for f in flights), default=0.0)
    cheapest_hotel = min((h.price_per_night for h in hotels), default=0.0)
    # Costo total de actividades por persona
    activities_per_person = sum(a.price_per_person for a in activities)

    nights = 0
    if state.get("arrival_date") and state.get("leave_date"):
        arrival = date.fromisoformat(state["arrival_date"])
        leave = date.fromisoformat(state["leave_date"])
        nights = max((leave - arrival).days, 0)

    total = calc_budget.invoke({
        "price_hotel": cheapest_hotel,
        "price_flight": cheapest_flight,
        "price_activities": activities_per_person,
        "persons": state.get("persons", 1),
        "nights": nights,
    })

    return {"budget_total": total}


def response_node(state: Context) -> dict:
    flights = state.get("flights") or []
    hotels = state.get("hotels") or []
    activities = state.get("activities") or []
    budget_total = state.get("budget_total")

    last_user_message = state["messages"][-1].content

    prompt = f"""
        Create a travel summary using the available information.

        User request:
        {last_user_message}

        Flights:
        {flights if flights else "No se buscaron vuelos."}

        Hotels:
        {hotels if hotels else "No se buscaron hoteles."}

        Activities:
        {activities if activities else "No se buscaron actividades."}

        Budget total:
        {f"MX${budget_total:,.2f}" if budget_total else "No calculado."}

        Requirements:
        - Respond in the user's language
        - Use markdown formatting with tables when presenting structured data
        - If some information is missing, mention it clearly but briefly
    """

    response = model.invoke([
        SystemMessage(
            content="""
                You are an expert travel planner assistant.

                Rules:
                - ALWAYS answer in the same language as the user.
                - If the user writes in Spanish, respond ONLY in Spanish.
                - Never mix languages.
                - Use markdown tables when presenting structured information (flights, hotels, activities).
                - Never invent information. Use ONLY data from the current context.
                - Be concise and professional.
            """
        ),
        HumanMessage(content=prompt),
    ])

    return {
        "messages": [AIMessage(content=response.content)]
    }


# =========================================================
# Router
# =========================================================

def route_after_extract(state: Context) -> list[str]:
    routes = []

    if state.get("wants_flights"):
        routes.append("flight_node")
    if state.get("wants_hotels"):
        routes.append("hotel_node")
    if state.get("wants_activities"):
        routes.append("activity_node")

    # Si no hay herramientas que invocar, ir directo a respuesta
    if not routes:
        return ["response_node"]

    return routes


def gather_node(state: Context) -> dict:
    """
    Nodo de sincronización: espera a que todos los tool nodes paralelos
    terminen antes de pasar a budget_node.
    No modifica el estado — solo sirve como punto de convergencia.
    """
    return {}


# =========================================================
# Graph
# =========================================================

agent_builder = StateGraph(Context)

agent_builder.add_node("extract_trip_info", extract_trip_info)
agent_builder.add_node("flight_node", flight_node)
agent_builder.add_node("hotel_node", hotel_node)
agent_builder.add_node("activity_node", activity_node)
agent_builder.add_node("gather_node", gather_node)
agent_builder.add_node("budget_node", budget_node)
agent_builder.add_node("response_node", response_node)

agent_builder.add_edge(START, "extract_trip_info")
agent_builder.add_conditional_edges("extract_trip_info", route_after_extract)

agent_builder.add_edge("flight_node", "gather_node")
agent_builder.add_edge("hotel_node", "gather_node")
agent_builder.add_edge("activity_node", "gather_node")
agent_builder.add_edge("gather_node", "budget_node")
agent_builder.add_edge("budget_node", "response_node")
agent_builder.add_edge("response_node", END)

agent = agent_builder.compile()