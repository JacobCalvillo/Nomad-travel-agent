from datetime import date, datetime
import logging

from agent.state import Context
from agent.tools import search_flights, search_hotels, get_activities, calc_budget
from agent.model import model, route_strategist, RouteStrategy

from langchain.messages import SystemMessage, AIMessage, HumanMessage

logger = logging.getLogger(__name__)


# =========================================================
# Route Strategy Node (The Strategist)
# =========================================================

def route_strategy_node(state: Context) -> dict:
    """Agente estratega: analiza origen/destino y decide si conviene vuelo directo o self-transfer."""
    
    origin = state.get("origin_iata") or state.get("origin", "?")
    destination = state.get("destination_iata") or state.get("place", "?")
    arrival_date = state.get("arrival_date", "?")

    system_prompt = """
        Eres un experto en Travel Hacking y optimización de rutas aéreas.
        Tu única misión es analizar si la ruta de origen a destino se puede hacer
        MÁS BARATA comprando dos boletos sencillos independientes (Self-Transfer)
        en vez de un vuelo directo o con escala de una sola aerolínea.

        Reglas de decisión:
        1. Si la distancia es corta (vuelo doméstico < 3 horas, ej. MTY-GDL, MTY-MEX), usa strategy='Direct'.
        2. Si es internacional a destinos cercanos (México-USA, México-Centroamérica), usa strategy='Direct'.
        3. Si el vuelo es intercontinental o de muy larga distancia (México a Asia, Europa, Oceanía, África),
           SIEMPRE evalúa la opción 'SelfTransfer'. El usuario quiere saber la forma MÁS BARATA,
           y comprar dos sencillos cruzando aerolíneas low-cost suele ser 40-60% más barato.
        4. Para ir a Asia desde México, los hubs ideales son: LAX (Los Ángeles), SFO (San Francisco), YVR (Vancouver).
        5. Para ir a Europa desde México, los hubs ideales son: MAD (Madrid), JFK (Nueva York), MIA (Miami).
        6. Para ir a Oceanía desde México, los hubs son: LAX (Los Ángeles), SYD (Sydney vía LAX).
        7. Propón máximo 2 hubs a evaluar para no desperdiciar créditos de API.
        8. Sé explícito en tu razonamiento sobre por qué es más barato el self-transfer.
    """
    
    user_message = f"""Analiza esta ruta:
    - Origen IATA: {origin}
    - Destino IATA: {destination}
    - Fecha de salida: {arrival_date}
    
    ¿Debo buscar vuelo directo o self-transfer para optimizar el costo?"""
    
    try:
        strategy: RouteStrategy = route_strategist.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        logger.info(f"ROUTE STRATEGY: {strategy.strategy} | Hubs: {[h.iata for h in strategy.hubs]} | Intl: {strategy.is_international}")
        logger.info(f"STRATEGY REASONING: {strategy.reasoning}")
        return {"route_strategy": strategy.model_dump()}
    except Exception as e:
        logger.error(f"Error en route_strategy_node: {e}")
        # Fallback: buscar directo
        return {"route_strategy": {"strategy": "Direct", "hubs": [], "reasoning": "Fallback", "is_international": False}}


def flight_node(state: Context) -> dict:
    logger.debug(f"FLIGHT NODE STATE: {state}")
    
    if not state.get("arrival_date"):
        return {
            "needs_clarification": True,
            "clarification_messages": [
                "¿Qué fecha de salida tienes en mente para el vuelo?"
            ]
        }

    if date.fromisoformat(state["arrival_date"]) < date.today():
        return {"messages": [AIMessage(content="La fecha de salida ya pasó. ¿Cuándo quieres viajar?")]}

    if not state.get("origin"):
        return {"messages": [AIMessage(content="¿Desde qué ciudad vas a salir?")]}

    if not state.get("place"):
        return {"messages": [AIMessage(content="¿A qué destino quieres viajar?")]}

    origin_code = state.get("origin_iata") or state.get("origin")
    destination_code = state.get("destination_iata") or state.get("place")
    arrival_date = state["arrival_date"]
    leave_date = state.get("leave_date")
    passengers = state.get("persons", 1)

    route_strategy = state.get("route_strategy") or {"strategy": "Direct", "hubs": []}
    strategy = route_strategy.get("strategy", "Direct")
    hubs = route_strategy.get("hubs", [])

    all_flights = []

    if strategy == "SelfTransfer" and hubs:
        logger.info(f"FLIGHT NODE: Ejecutando SelfTransfer con hubs: {[h['iata'] for h in hubs]}")
        
        for hub in hubs:
            hub_iata = hub["iata"]
            hub_city = hub.get("city", hub_iata)

            try:
                seg1 = search_flights.invoke({
                    "passengers": passengers, "origin": origin_code,
                    "destination": hub_iata, "arrival_date": arrival_date,
                    "leave_date": None, "type_of_flight": "Sencillo",
                })
            except Exception as e:
                logger.error(f"  Error tramo 1 ({origin_code}->{hub_iata}): {e}"); seg1 = []

            try:
                seg2 = search_flights.invoke({
                    "passengers": passengers, "origin": hub_iata,
                    "destination": destination_code, "arrival_date": arrival_date,
                    "leave_date": None, "type_of_flight": "Sencillo",
                })
            except Exception as e:
                logger.error(f"  Error tramo 2 ({hub_iata}->{destination_code}): {e}"); seg2 = []

            compatible_pairs = []
            for s1 in (seg1 or []):
                for s2 in (seg2 or []):
                    try:
                        arr_hub = datetime.fromisoformat(s1.arrival_time)
                        dep_dest = datetime.fromisoformat(s2.departure_time)
                        diff_hours = (dep_dest - arr_hub).total_seconds() / 3600
                        if 3 <= diff_hours <= 24:
                            compatible_pairs.append((s1, s2, diff_hours))
                    except Exception:
                        pass

            if compatible_pairs:
                best = min(compatible_pairs, key=lambda x: x[0].price + x[1].price)
                s1, s2, wait = best
                from agent.tools import SearchFlightsResponse
                all_flights.append(SearchFlightsResponse(
                    leg_type=f"Ida (via {hub_city}, {wait:.1f}h escala)",
                    airline=f"{s1.airline} + {s2.airline}",
                    price=s1.price + s2.price,
                    duration=f"{s1.duration} + {s2.duration}",
                    departure_time=s1.departure_time,
                    arrival_time=s2.arrival_time,
                    layovers=1
                ))
                logger.info(f"  Hub {hub_iata}: MXN ${s1.price + s2.price:.0f} | Espera: {wait:.1f}h")
            else:
                logger.info(f"  Hub {hub_iata}: Sin pares compatibles de horarios.")

        if leave_date:
            try:
                return_flights = search_flights.invoke({
                    "passengers": passengers, "origin": destination_code,
                    "destination": origin_code, "arrival_date": leave_date,
                    "leave_date": None, "type_of_flight": "Sencillo",
                })
                all_flights.extend(return_flights or [])
            except Exception as e:
                logger.error(f"Error en vuelo de regreso (SelfTransfer): {e}")

        if not all_flights:
            logger.info("SelfTransfer sin resultados. Fallback a vuelo directo.")
            strategy = "Direct"

    if strategy == "Direct":
        logger.info(f"FLIGHT NODE: Estrategia Directa ({origin_code} -> {destination_code})")
        direct_flights = search_flights.invoke({
            "passengers": passengers, "origin": origin_code,
            "destination": destination_code, "arrival_date": arrival_date,
            "leave_date": leave_date,
            "type_of_flight": "Redondo" if leave_date else "Sencillo",
        })
        all_flights.extend(direct_flights or [])

    if not all_flights:
        return {
            "flights": [],
            "messages": [AIMessage(content="No encontré vuelos para esa ruta y fecha. Prueba con otras fechas.")]
        }

    return {"flights": all_flights}


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

    # Extraer precios de Ida y Regreso de forma independiente
    ida_prices = [getattr(f, 'price', 0) for f in flights if getattr(f, 'leg_type', '') == 'Ida']
    regreso_prices = [getattr(f, 'price', 0) for f in flights if getattr(f, 'leg_type', '') == 'Regreso']
    
    cheapest_ida = min(ida_prices, default=0.0) if ida_prices else 0.0
    cheapest_regreso = min(regreso_prices, default=0.0) if regreso_prices else 0.0
    cheapest_flight = cheapest_ida + cheapest_regreso

    hotel_prices = [getattr(h, 'price_per_night', 0) for h in hotels if hasattr(h, 'price_per_night')]
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
        - VERY IMPORTANT: Flights are now provided as separate one-way options with 'leg_type' ("Ida" or "Regreso") and their exact true prices.
        - You MUST create two distinct markdown tables: one for "Opciones de Ida" and one for "Opciones de Regreso", showing up to the top 3 options for each leg.
        - The flight tables MUST include a column for 'Escalas' (Layovers) using the new 'layovers' field.
        - You MUST explicitly tell the user that the budget shown assumes they pick the absolute cheapest combination of flights, but that they have the flexibility to choose any of the other options from the tables based on their preferred schedule.
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