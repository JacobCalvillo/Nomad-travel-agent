import getpass
import os

from datetime import date

from agent.graph import agent
from langchain.messages import HumanMessage

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")


result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="""
                Busca vuelos y estancia para 2 personas
                de Monterrey a Oaxaca
                del 15 al 20 de junio
                con presupuesto de 15 mil pesos
                """
            )
        ]
    }
)

print(result["messages"][-1].content)

# for chunk in agent.stream(
#     {
#         "messages": [
#             HumanMessage(
#                 content="""
#                 Busca vuelos para 2 personas
#                 de Mexico City a Oaxaca
#                 del 15 al 20 de junio
#                 con presupuesto de 15 mil pesos
#                 """
#             )
#         ]
#     },
#     stream_mode=['updates','custom'],
#     version='v2'
# ):
#     if chunk['type'] == 'updates':
#         for node_name, state in chunk['data'].items():
#             print(f'Node {node_name} updated: {state}')
#     elif chunk['type'] == 'custom':
#         print(f'Status: {chunk['data']['status']}')




