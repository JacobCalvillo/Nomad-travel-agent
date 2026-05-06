import os
from datetime import date

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition

from langchain.messages import SystemMessage, ToolMessage, AIMessage
from langchain.chat_models import init_chat_model

from agent.tools import search_flights, search_hotels, get_activities, calc_budget
from agent.state import Context


class TripInfo(BaseModel):
    persons: int | str | None = None
    budget: float | str | None = None
    
    place: str | None = None
    origin: str | None = None
    
    arrival_date: str | None = None
    leave_date: str | None = None
    

def normalize_bool(value):

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() == "true"

    return False

model = init_chat_model(
    model=f'groq:{os.getenv('GROQ_MODEL')}'
)

extractor = model.with_structured_output(TripInfo)

model_with_tools = model.bind_tools([search_flights, search_hotels, get_activities, calc_budget])

def llm_call(state: Context):
    """LLM decides whether to call a tool or not"""

    if state.get('llm_calls', 0) > 5:
        return {
            'messages': [
                AIMessage("Maxtool iterations reached.")
            ]
        }
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="""
                        You are a professional vacation planner.

                        Rules:
                        - Always respond in the user's language.
                        - Never invent travel information.
                        - Use only information available in the current state.
                        - Use tools only when necessary.
                        - Present flight and hotel results in markdown tables.
                        - If required information is missing, ask concise follow-up questions.
                        - Never call tools repeatedly with the same parameters.
                        - Do not calculate budgets unless the user explicitly asks for total costs or budgeting.
                        - After completing the requested task, stop and provide the final answer immediately.
                        - Never suggest hotels, activities or budgeting unless explicitly requested.
                        """
                    ),
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

tools_by_name = {tool.name: tool for tool in [search_flights, search_hotels, get_activities, calc_budget]}


def extract_trip_info(state: Context):

    last_message = state["messages"][-1].content

    extracted = extractor.invoke(f"""
    Extract travel information from this message.

    Rules:
    - Return numbers as numbers, not strings
    - Dates must use YYYY-MM-DD format

    Today is {date.today()}.

    Message:
    {last_message}
    """)

    persons = extracted.persons
    budget = extracted.budget

    if isinstance(persons, str):
        persons = int(persons)

    if isinstance(budget, str):
        budget = float(budget)

    message = last_message.lower()

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
            "hoteles"
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

    arrival = extracted.arrival_date
    leave = extracted.leave_date

    if arrival and leave:

        arrival_date = date.fromisoformat(arrival)
        leave_date = date.fromisoformat(leave)

        if leave_date < arrival_date:
            arrival, leave = leave, arrival

    updates = {}

    if persons is not None:
        updates["persons"] = persons

    if budget is not None:
        updates["budget"] = budget

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

def tools(state: Context):

    result = []

    for tool_call in state['messages'][-1].tool_calls:

        tool = tools_by_name[tool_call['name']]

        args = dict(tool_call['args'])

        # -------------------------
        # Flights
        # -------------------------

        if tool.name == 'search_flights':

            args['arrival_date'] = state['arrival_date']
            args['leave_date'] = state['leave_date']
            args['passengers'] = state['persons']
            args['destination'] = state['place']
            args['current_location'] = state['origin']

        # -------------------------
        # Hotels
        # -------------------------

        if tool.name == 'search_hotels':

            if not state.get("wants_hotels"):

                result.append(
                    ToolMessage(
                        content="Hotel search was not requested.",
                        tool_call_id=tool_call['id']
                    )
                )

                continue

            args['place'] = state['place']
            args['check_in_date'] = state['arrival_date']
            args['check_out_date'] = state['leave_date']
            args['adults'] = state['persons']
            args['children'] = 0

        # -------------------------
        # Activities
        # -------------------------

        if tool.name == 'get_activities':

            if not state.get("wants_activities"):

                result.append(
                    ToolMessage(
                        content="Activities search was not requested.",
                        tool_call_id=tool_call['id']
                    )
                )

                continue

            args['place'] = state['place']

        # -------------------------
        # Execute tool
        # -------------------------

        print("TOOL:", tool.name)
        print("ARGS:", args)

        try:
            observation = tool.invoke(args)

        except Exception as e:

            observation = f"Tool error: {str(e)}"

        result.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call['id']
            )
        )

    return {"messages": result}

agent_builder = StateGraph(Context)

agent_builder.add_node('llm_call', llm_call)
agent_builder.add_node('extract_trip_info', extract_trip_info)
agent_builder.add_node('tools', tools)

agent_builder.add_conditional_edges('llm_call', tools_condition)
agent_builder.add_edge(START, 'extract_trip_info')
agent_builder.add_edge('extract_trip_info', 'llm_call')
agent_builder.add_edge('tools', 'llm_call')

agent = agent_builder.compile()
