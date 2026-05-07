import os
from datetime import date

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition

from langchain.messages import (
    SystemMessage,
    ToolMessage,
    AIMessage
)

from langchain.chat_models import init_chat_model

from agent.tools import (
    search_flights,
    search_hotels,
    get_activities,
    calc_budget
)

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


# =========================================================
# Models
# =========================================================

model = init_chat_model(
    model=f"groq:{os.getenv('GROQ_MODEL')}"
)

extractor = model.with_structured_output(
    TripInfo
)

model_with_tools = model.bind_tools(
    [
        search_flights,
        search_hotels,
        get_activities,
        calc_budget
    ]
)


# =========================================================
# LLM Node
# =========================================================

def llm_call(state: Context):

    if state.get("llm_calls", 0) > 5:

        return {
            "messages": [
                AIMessage(
                    content="Max tool iterations reached."
                )
            ]
        }

    response = model_with_tools.invoke(
        [
            SystemMessage(
                content=f"""
            You are a professional vacation planner.

            Flights:
            {state.get("flights")}

            Hotels:
            {state.get("hotels")}

            Activities:
            {state.get("activities")}

            Budget:
            {state.get("budget")}

            Rules:
            - Never say data is unavailable if data exists
            - Use markdown tables
            - Use ONLY the data provided
            - If flights already exist in the state, do not call search_flights again
            - If hotels already exist in the state, do not call search_hotels again
            - If activities already exist in the state, do not call get_activities again
            - If enough information exists to answer the user, answer directly
            """
            )
        ]
        + state["messages"]
    )

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================================================
# Extractor Node
# =========================================================

def extract_trip_info(state: Context):

    last_message = state["messages"][-1].content

    extracted = extractor.invoke(
            f"""
            Extract travel information from this message.

            Rules:
            - Return numbers as numbers
            - Dates must use YYYY-MM-DD format

            Today is {date.today()}.

            Message:
            {last_message}
            """
    )

    message = last_message.lower()

    # ------------------------------------
    # Intent detection
    # ------------------------------------

    wants_flights = any(
        word in message
        for word in [
            "vuelo",
            "vuelos",
            "flight",
            "flights"
        ]
    )

    wants_hotels = any(
        word in message
        for word in [
            "hotel",
            "hoteles",
            "estancia",
            "hospedaje",
            "alojamiento",
            "airbnb"
        ]
    )

    wants_activities = any(
        word in message
        for word in [
            "actividad",
            "actividades",
            "hacer",
            "things to do"
        ]
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

    updates["wants_flights"] = wants_flights
    updates["wants_hotels"] = wants_hotels
    updates["wants_activities"] = wants_activities

    print("EXTRACTED:", updates)

    return updates


# =========================================================
# Tools Node
# =========================================================

tools_by_name = {
    tool.name: tool
    for tool in [
        search_flights,
        search_hotels,
        get_activities,
        calc_budget
    ]
}


def tools(state: Context):

    result = []
    updates = {}

    for tool_call in state["messages"][-1].tool_calls:

        tool = tools_by_name[
            tool_call["name"]
        ]

        # =================================================
        # Flights
        # =================================================

        if tool.name == "search_flights":

            args = {
                "passengers": state.get('persons', 1),
                "origin": state["origin"],
                "destination": state["place"],
                "arrival_date": state["arrival_date"],
                "leave_date": state["leave_date"],
                "type_of_flight": "Redondo"
            }

        # =================================================
        # Hotels
        # =================================================

        elif tool.name == "search_hotels":

            if not state.get("wants_hotels"):

                result.append(
                    ToolMessage(
                        content="Hotel search was not requested.",
                        tool_call_id=tool_call["id"]
                    )
                )

                continue

            args = {
                "place": state["place"],
                "check_in_date": state["arrival_date"],
                "check_out_date": state["leave_date"],
                "adults": state.get('persons', 1),
                "children": state.get("children", 0)
            }

        # =================================================
        # Activities
        # =================================================

        elif tool.name == "get_activities":

            if not state.get("wants_activities"):

                result.append(
                    ToolMessage(
                        content="Activities search was not requested.",
                        tool_call_id=tool_call["id"]
                    )
                )

                continue

            args = {
                "place": state["place"]
            }

        # =================================================
        # Budget
        # =================================================

        elif tool.name == "calc_budget":

            flights = state.get("flights", [])
            hotels = state.get("hotels", [])
            activities = state.get("activities", [])

            cheapest_flight = (
                min(f.price for f in flights)
                if flights
                else 0
            )

            cheapest_hotel = (
                min(
                    h.price_per_night
                    for h in hotels
                    if h.price_per_night
                )
                if hotels
                else 0
            )

            activities_total = (
                sum(a.price_per_person for a in activities)
                if activities
                else 0
            )

            arrival = date.fromisoformat(
                state["arrival_date"]
            )

            leave = date.fromisoformat(
                state["leave_date"]
            )

            nights = (leave - arrival).days

            args = {
                "price_hotel": cheapest_hotel,
                "price_flight": cheapest_flight,
                "price_activities": activities_total,
                "persons": state.get('persons', 1),
                "nights": nights
            }

        else:
            continue

        print("TOOL:", tool.name)
        print("ARGS:", args)

        try:

            observation = tool.invoke(args)

            # =====================================
            # SAVE RESULTS INTO STATE
            # =====================================

            if tool.name == "search_flights":
                updates["flights"] = observation

            elif tool.name == "search_hotels":
                updates["hotels"] = observation

            elif tool.name == "get_activities":
                updates["activities"] = observation
                

        except Exception as e:

            print("TOOL ERROR:", e)

            observation = f"Tool error: {str(e)}"

        # =========================================
        # CLEAN TOOL MESSAGE
        # =========================================

        if isinstance(observation, list):

            content = "\n".join(
                [
                    item.model_dump_json(indent=2)
                    for item in observation
                ]
            )

        else:
            content = str(observation)

        result.append(
            ToolMessage(
                content=content,
                tool_call_id=tool_call["id"]
            )
        )

    updates["messages"] = result

    return updates


# =========================================================
# Graph
# =========================================================

agent_builder = StateGraph(Context)

agent_builder.add_node(
    "extract_trip_info",
    extract_trip_info
)

agent_builder.add_node(
    "llm_call",
    llm_call
)

agent_builder.add_node(
    "tools",
    tools
)

agent_builder.add_edge(
    START,
    "extract_trip_info"
)

agent_builder.add_edge(
    "extract_trip_info",
    "llm_call"
)

agent_builder.add_conditional_edges(
    "llm_call",
    tools_condition
)

agent_builder.add_edge(
    "tools",
    "llm_call"
)

agent = agent_builder.compile()