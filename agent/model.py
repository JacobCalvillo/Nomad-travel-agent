import os

from typing import Optional
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
    origin: Optional[str] = Field(None, description="Ciudad origen")
    arrival_date: Optional[str] = Field(None, description="Fecha de llegada (YYYY-MM-DD)")
    leave_date: Optional[str] = Field(None, description="Fecha de regreso (YYYY-MM-DD)")
    budget: Optional[float] = Field(None, description="Presupuesto máximo del usuario")

    # Intenciones (Decididas por el LLM, no por código)
    wants_flights: bool = Field(False, description="True si el usuario necesita buscar vuelos o si se infiere por el contexto de un viaje nuevo.")
    wants_hotels: bool = Field(False, description="True si el usuario necesita hospedaje.")
    wants_activities: bool = Field(False, description="True si el usuario pregunta por tours, actividades o qué hacer.")

model = init_chat_model(model=f"google_genai:{os.getenv('GEMINI_MODEL')}")

extractor = model.with_structured_output(TripInfo)