import os
from datetime import date

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain.messages import SystemMessage, AIMessage, HumanMessage

from agent.tools import search_flights,search_hotels,get_activities,calc_budget
from agent.state import Context


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

model = init_chat_model(model=f"groq:{os.getenv('GROQ_MODEL')}")

extractor = model.with_structured_output(TripInfo)

# =========================================================
# Extractor Node
# =========================================================

def extract_trip_info(state: Context):
    last_message = state["messages"][-1].content
    
    try:
        extracted = extractor.invoke(
                f"""
                Extract travel information from this message.
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

                User message: {last_message}
                """
            )
    except Exception:
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
    # Fix inverted dates
    # ------------------------------------

    arrival = extracted.arrival_date
    leave = extracted.leave_date

    if arrival and leave:
        arrival_date = date.fromisoformat(arrival)
        leave_date = date.fromisoformat(leave)

        if leave_date < arrival_date:
            arrival, leave = leave, arrival

    updates = {}

    # ------------------------------------
    # Persist extracted fields
    # ------------------------------------

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

    print("EXTRACTED:", updates)

    return updates

def response_node(state: Context):
    flights = state.get("flights", [])
    hotels = state.get("hotels", [])
    activities = state.get("activities", [])
    budget_total = state.get("budget_total")

    last_user_message = state['messages'][-1].content
    
    prompt = f"""
                Create a travel summary using the available information.

                User request:
                {last_user_message}
                
                Flights:
                {flights}

                Hotels:
                {hotels}

                Activities:
                {activities}
                
                Budget total:
                {budget_total}

                Requirements:
                - Respond in the user's language
                - Use markdown formatting
                - If some information is missing, mention it clearly
            """
    
    response = model.invoke([
        SystemMessage(
            content="""
                    You are an expert travel planner assistant.

                    Rules:
                    - ALWAYS answer in the same language as the user.
                    - If the user writes in Spanish, respond ONLY in Spanish.
                    - Never mix languages.
                    - Use markdown tables when presenting structured information.
                    - Never invent information.
                    - Use ONLY the information provided in the current context.
                    - Be concise and professional.
                    
                    """
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "messages": [
            AIMessage(content=response.content)
        ]
    }

def flight_node(state: Context):
    if not state.get('arrival_date'):
        return {
            'messages': [AIMessage(content="Que fechas deseas para el vuelo?")]
        }
    
    flights = search_flights.invoke({
        "passengers": state["persons"],
        "origin": state["origin"],
        "destination": state["place"],
        "arrival_date": state["arrival_date"],
        "leave_date": state.get('leave_date'),
        "type_of_flight": "Redondo"
    })

    return {
        "flights": flights
    }
    
def hotel_node(state: Context):
    if not state.get("arrival_date") or not state.get("leave_date"):
        return {
            "messages": [AIMessage(content="¿Qué fechas deseas para el hospedaje?")]
        }
    
    hotels = search_hotels.invoke({
        "place": state["place"],
        "check_in_date": state["arrival_date"],
        "check_out_date": state["leave_date"],
        "adults": state["persons"],
        "children": state.get("children", 0)
    })

    return {
        "hotels": hotels
    }

def activity_node(state: Context):
    activities = get_activities.invoke({"place": state["place"]})

    return {
        "activities": activities
    }
    

def budget_node(state:Context):
    flights = state.get("flights", [])
    hotels = state.get("hotels", [])
    activities = state.get("activities", [])

    cheapest_flight = min([f.price for f in flights],default=0)
    cheapest_hotel = min([h.price_per_night for h in hotels],default=0)
    activities_total = sum(a.price_per_person for a in activities) if activities else 0

    nights = 0

    if state.get("arrival_date") and state.get("leave_date"):
        arrival = date.fromisoformat(state["arrival_date"])
        leave = date.fromisoformat(state["leave_date"])
        nights = (leave - arrival).days

    total = calc_budget.invoke({
        "price_hotel": cheapest_hotel,
        "price_flight": cheapest_flight,
        "price_activities": activities_total,
        "persons": state.get("persons", 1),
        "nights": nights
    })

    return {
        "budget_total": total
    }

def route_after_extract(state: Context):
    routes = []
    
    if state.get('wants_flights'):
        routes.append("flight_node")
        
    if state.get('wants_hotels'):
        routes.append('hotel_node')
    
    if state.get('wants_activities'):
        routes.append('activity_node')
    
    if not routes:
        return ['response_node']
    
    return routes

def gather_node(state: Context):
    return {}

# =========================================================
# Graph
# =========================================================

agent_builder = StateGraph(Context)

agent_builder.add_node("extract_trip_info",extract_trip_info)
agent_builder.add_node("flight_node", flight_node)
agent_builder.add_node("hotel_node", hotel_node)
agent_builder.add_node("activity_node", activity_node)
agent_builder.add_node('budget_node', budget_node)
agent_builder.add_node('gather_node', gather_node)
agent_builder.add_node("response_node",response_node)

agent_builder.add_edge(START,"extract_trip_info")
agent_builder.add_conditional_edges("extract_trip_info",route_after_extract)

agent_builder.add_edge("flight_node","gather_node")
agent_builder.add_edge("hotel_node","gather_node")
agent_builder.add_edge("activity_node","gather_node")
agent_builder.add_edge("gather_node", "budget_node")
agent_builder.add_edge("budget_node", "response_node")
agent_builder.add_edge('response_node', END)



agent = agent_builder.compile()