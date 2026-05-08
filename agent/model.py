import os

from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv


load_dotenv()

# =========================================================
# Structured extraction schema
# =========================================================


class TripInfo(BaseModel):
    """Información estructurada del viaje y análisis de intención."""
    
    # Razonamiento
    reasoning: str = Field(description="Breve explicación de por qué se extrajeron estos datos y qué intenciones se detectaron.")
    
    # Datos
    persons: Optional[int] = Field(None, description="Número de adultos")
    children: Optional[int] = Field(None, description="Número de niños")
    place: Optional[str] = Field(None, description="Ciudad destino")
    destination_iata: Optional[str] = Field(None, description="Código IATA de 3 letras del aeropuerto destino (ej. MEX, JFK, MAD). Indispensable para vuelos.")
    origin: Optional[str] = Field(None, description="Ciudad origen")
    origin_iata: Optional[str] = Field(None, description="Código IATA de 3 letras del aeropuerto de origen (ej. MEX, CUN). Indispensable para vuelos.")
    arrival_date: Optional[str] = Field(None, description="Fecha de llegada (YYYY-MM-DD)")
    leave_date: Optional[str] = Field(None, description="Fecha de regreso (YYYY-MM-DD)")
    budget: Optional[float] = Field(None, description="Presupuesto máximo del usuario")

    # Intenciones (Decididas por el LLM, no por código)
    wants_flights: bool = Field(False, description="True si el usuario necesita buscar vuelos o si se infiere por el contexto de un viaje nuevo.")
    wants_hotels: bool = Field(False, description="True si el usuario necesita hospedaje.")
    wants_activities: bool = Field(False, description="True si el usuario pregunta por tours, actividades o qué hacer.")


# =========================================================
# Route Strategy schema
# =========================================================

class FlightHub(BaseModel):
    """Un hub de conexión propuesto para el self-transfer."""
    iata: str = Field(description="Código IATA del aeropuerto hub (ej. LAX, SFO, JFK)")
    city: str = Field(description="Nombre de la ciudad del hub (ej. Los Ángeles)")
    reasoning: str = Field(description="Breve justificación de por qué este hub es conveniente")


class RouteStrategy(BaseModel):
    """Estrategia de ruta determinada por el agente estratega."""
    strategy: Literal["Direct", "SelfTransfer"] = Field(
        description=(
            "'Direct': Buscar vuelos directos/con escalas de la aerolínea entre origen y destino. "
            "'SelfTransfer': La ruta es más barata comprando dos boletos sencillos independientes vía un hub."
        )
    )
    hubs: list[FlightHub] = Field(
        default_factory=list,
        description="Lista de hubs a evaluar (solo si strategy=SelfTransfer). Máximo 2."
    )
    reasoning: str = Field(description="Explicación de la estrategia elegida y el análisis de costo-beneficio.")
    is_international: bool = Field(description="True si el vuelo cruza fronteras o continentes.")


model = init_chat_model(model=f"google_genai:{os.getenv('GEMINI_MODEL')}")

extractor = model.with_structured_output(TripInfo)
route_strategist = model.with_structured_output(RouteStrategy)