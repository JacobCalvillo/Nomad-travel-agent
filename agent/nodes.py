from datetime import date
import logging

from agent.state import Context
from agent.tools import search_flights, search_hotels, get_activities, calc_budget
from agent.model import model

from langchain.messages import SystemMessage, AIMessage, HumanMessage

logger = logging.getLogger(__name__)

def flight_node(state: Context) -> dict:
    logger.debug(f"FLIGHT NODE STATE: {state}")
    
    if not state.get("arrival_date"):
        return {
            "needs_clarification": True,
            "clarification_messages": [
                "¿Qué fecha de salida tienes en mente para el vuelo?"
            ]
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
    
    logger.debug(f"HOTEL NODE STATE: {state}")
    
    if not state.get("arrival_date") or not state.get("leave_date"):
        return {
            "needs_clarification": True,
            "clarification_messages": [
                "¿Qué fechas necesitas para el hospedaje? (entrada y salida)"
            ]
        }

    if not state.get("place"):
        return {
            "clarification_messages": [
                "¿En qué ciudad buscamos hotel?"
            ]
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
            "needs_clarification": True,
            "clarification_messages": [
                "¿En qué ciudad buscamos actividades?"
            ]
        }

    activities = get_activities.invoke({"place": state["place"]})
    return {"activities": activities}


def budget_node(state: Context) -> dict:
    flights = state.get("flights") or []
    hotels = state.get("hotels") or []
    activities = state.get("activities") or []

    # Extraemos precios asegurando que sean float o int
    flight_prices = [getattr(f, 'price', 0) for f in flights if hasattr(f, 'price')]
    hotel_prices = [getattr(h, 'price_per_night', 0) for h in hotels if hasattr(h, 'price_per_night')]
    
    cheapest_flight = min(flight_prices, default=0.0)
    cheapest_hotel = min(hotel_prices, default=0.0)
    
    activities_per_person = sum(getattr(a, 'price_per_person', 0) for a in activities)

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

    # =========================================================
    # Clarifications first
    # =========================================================

    clarification_messages = state.get("clarification_messages") or []

    if clarification_messages:
        unique_messages = list(dict.fromkeys(clarification_messages))
        return {
            "messages": [
                AIMessage(
                    content="\n".join(unique_messages)
                )
            ]
        }

    # =========================================================
    # Normal response flow
    # =========================================================

    flights = state.get("flights") or []
    hotels = state.get("hotels") or []
    activities = state.get("activities") or []
    budget_total = state.get("budget_total")

    # Calcular noches para darle contexto explícito al LLM
    nights = 0
    arrival_str = state.get("arrival_date")
    leave_str = state.get("leave_date")
    if arrival_str and leave_str:
        from datetime import date
        nights = max((date.fromisoformat(leave_str) - date.fromisoformat(arrival_str)).days, 0)

    last_user_message = state["messages"][-1].content

    prompt = f"""
        Create a travel summary using the available information.

        User request:
        {last_user_message}

        Trip Details (Extracted Context):
        - Origin: {state.get("origin", "N/A")}
        - Destination: {state.get("place", "N/A")}
        - Dates: {arrival_str} to {leave_str} (Total: {nights} nights)
        - Travelers: {state.get("persons", 1)} adults, {state.get("children", 0)} children

        Flights:
        {flights if flights else "No flights found."}

        Hotels:
        {hotels if hotels else "No hotels found."}

        Activities:
        {activities if activities else "No activities found."}

        Budget total:
        {f"MX${budget_total:,.2f}" if budget_total else "Not calculated."}

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
        "messages": [
            AIMessage(content=response.content)
        ]
    }

def sanity_check_node(state: Context) -> dict:
    errors = []
    today = date.today()
    
    arrival_str = state.get("arrival_date")
    leave_str = state.get("leave_date")
    
    # Validación de fechas base
    if arrival_str:
        arrival_dt = date.fromisoformat(arrival_str)
        if arrival_dt < today:
            errors.append(f"La fecha de llegada ({arrival_str}) ya pasó. Por favor, elige una fecha futura.")
            
    if arrival_str and leave_str:
        arrival_dt = date.fromisoformat(arrival_str)
        leave_dt = date.fromisoformat(leave_str)
        if leave_dt < arrival_dt:
            errors.append("La fecha de regreso no puede ser anterior a la de salida.")

    # Validación de dependencias de herramientas
    if state.get("wants_flights"):
        if not state.get("origin"):
            errors.append("Necesito saber de qué ciudad sales para buscar los vuelos.")
        if not state.get("place"):
            errors.append("¿A qué ciudad quieres viajar? (Me falta el destino).")
            
    if state.get("wants_hotels") and not state.get("leave_date"):
        errors.append("Para buscar hoteles necesito saber cuántas noches te quedarás (fecha de regreso).")

    # Si hay errores, activamos la bandera de clarificación
    if errors:
        return {
            "needs_clarification": True,
            "clarification_messages": errors
        }
    
    return {"needs_clarification": False, "clarification_messages": []}