import os
import logging
from datetime import date

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langchain.messages import SystemMessage

from agent.state import Context
from agent.model import extractor, TripInfo
from agent.nodes import sanity_check_node, flight_node, hotel_node, activity_node, budget_node, response_node, route_strategy_node
from agent.nodes import sanity_check_node

logger = logging.getLogger(__name__)


# =========================================================
# Extractor Node
# =========================================================

def extract_trip_info(state: Context) -> dict:
    recent_messages = state.get("messages", [])[-6:]
    
    system_prompt = f"""
    Eres el núcleo de razonamiento de Nomad. Tu misión es extraer datos y DECIDIR qué herramientas activar.
    
    Reglas de Decisión:
    1. Si el usuario menciona un destino y un origen, asume que quiere VUELOS (wants_flights=True) a menos que diga lo contrario.
    2. Si menciona quedarse varias noches o el destino es lejos, asume que quiere HOTEL (wants_hotels=True).
    3. Si el usuario dice "solo vuelo", pon wants_hotels=False.
    4. Si el usuario pregunta "qué hay de bueno allá" o "tours", pon wants_activities=True.
    
    Reglas CRÍTICAS para códigos IATA (origin_iata / destination_iata):
    - Debes emitir SIEMPRE el código IATA de un AEROPUERTO real, NO un código metropolitano.
    - Los códigos metropolitanos (TYO, NYC, LON, PAR, etc.) NO son aeropuertos y causarán errores.
    - Ejemplos CORRECTOS:
        * Tokio → NRT (Narita) o HND (Haneda)
        * Nueva York → JFK, LGA o EWR
        * Londres → LHR (Heathrow) o LGW (Gatwick)
        * París → CDG (Charles de Gaulle)
        * Seúl → ICN (Incheon)
        * Pekín → PEK (Capital) o PKX (Daxing)
        * Osaka → KIX (Kansai)
        * Monterrey → MTY
        * Ciudad de México → MEX
    - Si el destino es una ciudad con múltiples aeropuertos, elige el aeropuerto principal o más concurrido.
    
    Contexto temporal: Hoy es {date.today()}.
    Estado actual del viaje (úsalo para rellenar vacíos):
    - Origen: {state.get("origin")}
    - Destino: {state.get("place")}
    - Fechas: {state.get("arrival_date")} a {state.get("leave_date")}
    """

    try:
        # Aquí ocurre la magia del Structured Output
        extracted: TripInfo = extractor.invoke([
            SystemMessage(content=system_prompt),
            *recent_messages 
        ])
        
        # Validaciones de lógica de negocio (fechas)
        updates = extracted.model_dump(exclude_none=True)
        
        # Lógica para evitar sobreescribir con False si ya teníamos True (Persistencia de intención)
        for flag in ["wants_flights", "wants_hotels", "wants_activities"]:
            if not updates.get(flag) and state.get(flag):
                updates[flag] = True
             
        logger.debug(f"Razonamiento del Agente: {extracted.reasoning}")
        return updates

    except Exception as e:
        logger.error(f"Error en extracción: {e}")
        return {}


# =========================================================
# Tool Nodes
# =========================================================




# =========================================================
# Router
# =========================================================

def route_after_extract(state: Context) -> str:
    return "sanity_check_node"

def route_after_sanity_check(state: Context) -> list[str]:
    # Si el validador encontró errores, saltamos las herramientas
    if state.get("needs_clarification"):
        return ["response_node"]
    
    # Si quiere vuelos, primero pasamos por el estratega de rutas
    if state.get("wants_flights"):
        return ["route_strategy_node"]
    
    # Si no quiere vuelos, ruteamos directamente a hoteles/actividades
    routes = []
    if state.get("wants_hotels"):
        routes.append("hotel_node")
    if state.get("wants_activities"):
        routes.append("activity_node")
        
    return routes if routes else ["response_node"]


def route_after_strategy(state: Context) -> list[str]:
    """Después del estratega, lanza las búsquedas en paralelo."""
    routes = ["flight_node"]
    if state.get("wants_hotels"):
        routes.append("hotel_node")
    if state.get("wants_activities"):
        routes.append("activity_node")
    return routes

def gather_node(state: Context) -> dict:
    """
    Nodo de sincronización: espera a que todos los tool nodes paralelos
    terminen antes de pasar a budget_node.
    No modifica el estado — solo sirve como punto de convergencia.
    """
    return {}

def route_after_gather(state: Context) -> str:
    if state.get("needs_clarification"):
        return "response_node"

    return "budget_node"

# =========================================================
# Graph
# =========================================================

agent_builder = StateGraph(Context)

agent_builder.add_node("extract_trip_info", extract_trip_info)
agent_builder.add_node("sanity_check_node", sanity_check_node)
agent_builder.add_node("route_strategy_node", route_strategy_node)
agent_builder.add_node("flight_node", flight_node)
agent_builder.add_node("hotel_node", hotel_node)
agent_builder.add_node("activity_node", activity_node)
agent_builder.add_node("gather_node", gather_node)
agent_builder.add_node("budget_node", budget_node)
agent_builder.add_node("response_node", response_node)

agent_builder.add_edge(START, "extract_trip_info")
agent_builder.add_edge("extract_trip_info", "sanity_check_node")

agent_builder.add_conditional_edges("sanity_check_node", route_after_sanity_check)

# El estratega siempre lanza flight_node + hotel/activities en paralelo
agent_builder.add_conditional_edges("route_strategy_node", route_after_strategy)

agent_builder.add_edge("flight_node", "gather_node")
agent_builder.add_edge("hotel_node", "gather_node")
agent_builder.add_edge("activity_node", "gather_node")

agent_builder.add_conditional_edges("gather_node", route_after_gather)

agent_builder.add_edge("budget_node", "response_node")
agent_builder.add_edge("response_node", END)

agent = agent_builder.compile()