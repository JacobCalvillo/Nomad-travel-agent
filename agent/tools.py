import os
import httpx
from datetime import date
import airportsdata

from pydantic import BaseModel, Field

from agent.model import model
from langchain.messages import SystemMessage, HumanMessage
from langchain.tools import tool


# =========================================================
# Response schemas
# =========================================================

class SearchFlightsResponse(BaseModel):
    """Resultado de búsqueda de vuelos."""
    leg_type: str = Field(description="Tipo de trayecto ('Ida' o 'Regreso')")
    airline: str = Field(description="Nombre de la aerolínea (combinadas si hay escalas)")
    price: float = Field(default=0.0, description="Precio del trayecto en MXN")
    duration: str = Field(description="Duración total del viaje")
    departure_time: str = Field(description="Hora de salida (origen inicial)")
    arrival_time: str = Field(description="Hora de llegada (destino final)")
    layovers: int = Field(default=0, description="Cantidad de escalas")


class GetActivitiesResponse(BaseModel):
    """Actividad turística disponible en el destino."""
    name: str = Field(description="Nombre de la actividad")
    price_per_person: float = Field(description="Precio por persona en MXN")
    description: str = Field(description="Descripción de la actividad")
    category: str = Field(description="Categoría de la actividad")
    duration: str = Field(description="Duración de la actividad")


class SearchHotelsResponse(BaseModel):
    """Resultado de búsqueda de hoteles."""
    name: str = Field(description="Nombre del hotel")
    price_per_night: float = Field(description="Precio por noche en MXN")
    stars: float = Field(description="Calificación del hotel (1-5)")
    amenities: list[str] = Field(description="Amenidades disponibles")


_airports = airportsdata.load("IATA")


import unicodedata

def _get_airport_code(city: str) -> str:
    city_clean = city.upper().strip()
    if len(city_clean) == 3 and city_clean in _airports:
        return city_clean

    city_lower = city.lower().strip()

    for code, airport in _airports.items():
        if airport.get("city", "").lower() == city_lower:
            return code

    raise ValueError(
        f"No se encontró aeropuerto IATA para la ciudad: '{city}'. "
        "Verifica el nombre o prueba con la ciudad principal más cercana."
    )


# =========================================================
# Tools
# =========================================================

@tool("search_flights", description="Busca vuelos disponibles entre dos ciudades para las fechas indicadas.")
def search_flights(
    passengers: int | str,
    origin: str,
    destination: str,
    arrival_date: date | str,
    leave_date: date | str | None,
    type_of_flight: str = "Redondo",
) -> list[SearchFlightsResponse]:
    if isinstance(passengers, str):
        passengers = int(passengers)

    if isinstance(arrival_date, str):
        arrival_date = date.fromisoformat(arrival_date)
    if leave_date and isinstance(leave_date, str):
        leave_date = date.fromisoformat(leave_date)

    flight_type_map = {"Redondo": "1", "Sencillo": "2", "Paradas": "3"}
    type_code = flight_type_map.get(type_of_flight, "1")

    departure_code = _get_airport_code(origin)
    arrival_code = _get_airport_code(destination)

    print(f"SEARCH_FLIGHTS: {origin} ({departure_code}) → {destination} ({arrival_code})")

    params_outbound = {
        "engine": "google_flights",
        "departure_id": departure_code,
        "arrival_id": arrival_code,
        "currency": "MXN",
        "type": "2",  # Sencillo
        "outbound_date": arrival_date.isoformat(),
        "api_key": os.getenv("SER_API_API_KEY"),
        "adults": passengers,
        "no_cache": "true",
        "gl": "mx",
        "hl": "es"
    }

    flights: list[SearchFlightsResponse] = []

    try:
        r_out = httpx.get("https://serpapi.com/search", params=params_outbound, timeout=15.0)
        r_out.raise_for_status()
        raw_out = r_out.json().get("best_flights", []) or r_out.json().get("other_flights", [])
        
        for option in raw_out[:3]:  # Top 3 de Ida
            first_leg = option["flights"][0]
            last_leg = option["flights"][-1]
            airline = first_leg["airline"] if len(option["flights"]) == 1 else f"{first_leg['airline']} (Múltiples)"
            flights.append(
                SearchFlightsResponse(
                    leg_type="Ida",
                    airline=airline,
                    price=float(option.get("price", 0)),
                    duration=f"{option.get('total_duration', 0)} min",
                    departure_time=first_leg.get("departure_airport", {}).get("time", ""),
                    arrival_time=last_leg.get("arrival_airport", {}).get("time", ""),
                    layovers=len(option.get("layovers", []))
                )
            )
    except Exception as e:
        print(f"Error en vuelo de ida: {e}")

    if leave_date:
        params_inbound = params_outbound.copy()
        params_inbound["departure_id"] = arrival_code
        params_inbound["arrival_id"] = departure_code
        params_inbound["outbound_date"] = leave_date.isoformat()
        
        try:
            r_in = httpx.get("https://serpapi.com/search", params=params_inbound, timeout=15.0)
            r_in.raise_for_status()
            raw_in = r_in.json().get("best_flights", []) or r_in.json().get("other_flights", [])
            
            for option in raw_in[:3]:  # Top 3 de Regreso
                first_leg = option["flights"][0]
                last_leg = option["flights"][-1]
                airline = first_leg["airline"] if len(option["flights"]) == 1 else f"{first_leg['airline']} (Múltiples)"
                flights.append(
                    SearchFlightsResponse(
                        leg_type="Regreso",
                        airline=airline,
                        price=float(option.get("price", 0)),
                        duration=f"{option.get('total_duration', 0)} min",
                        departure_time=first_leg.get("departure_airport", {}).get("time", ""),
                        arrival_time=last_leg.get("arrival_airport", {}).get("time", ""),
                        layovers=len(option.get("layovers", []))
                    )
                )
        except Exception as e:
            print(f"Error en vuelo de regreso: {e}")

    print(f"FLIGHTS encontrados: {len(flights)} (Top 3 por trayecto)")
    return flights


@tool("search_hotels", description="Busca hoteles disponibles en el destino para las fechas indicadas.")
def search_hotels(
    place: str,
    check_in_date: str,
    check_out_date: str,
    adults: int,
    children: int,
) -> list[SearchHotelsResponse]:
    params = {
        "engine": "google_hotels",
        "q": f"Hotels in {place}",
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": adults,
        "children": children,
        "currency": "MXN",
        "gl": "mx",
        "hl": "es",
        "no_cache": "true",
        "api_key": os.getenv("SER_API_API_KEY"),
    }

    if children > 0:
        params["children_ages"] = ",".join(["5"] * children)

    try:
        r = httpx.get("https://serpapi.com/search", params=params, timeout=15.0)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        print(f"SERPAPI ERROR hoteles: {e.response.status_code} — {e.response.text}")
        return []

    response = r.json()
    hotels: list[SearchHotelsResponse] = []

    for hotel in response.get("properties", [])[:5]:
        price_str = hotel.get("rate_per_night", {}).get("lowest", "0")
        price = float(
            price_str
            .replace("MX$", "")
            .replace("$", "")
            .replace(",", "")
            .replace('\u202f', "")
            .strip() or 0
        )
        currency = response.get("search_parameters", {}).get("currency", "USD")
        if currency == "USD":
            price = price * 17.5  
    
        hotels.append(
            SearchHotelsResponse(
                name=hotel.get("name", "Sin nombre"),
                price_per_night=price,
                stars=float(hotel.get("overall_rating") or 0),
                amenities=hotel.get("amenities", []),
            )
        )


    print(f"HOTELS encontrados: {len(hotels)}")
    return hotels


class ActivityPriceEstimate(BaseModel):
    name: str = Field(description="Nombre de la actividad")
    estimated_price_mxn: float = Field(description="Costo estimado de entrada en MXN")

class ActivityPrices(BaseModel):
    prices: list[ActivityPriceEstimate]

@tool("get_activities", description="Obtiene actividades turísticas disponibles en el destino.")
def get_activities(place: str) -> list[GetActivitiesResponse]:
    params = {
        "engine": "google_local",
        "q": f"Things to do in {place}",
        "api_key": os.getenv("SER_API_API_KEY"),
        "hl": "es",
        "gl": "mx"
    }

    try:
        r = httpx.get("https://serpapi.com/search", params=params, timeout=15.0)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        print(f"SERPAPI ERROR actividades: {e.response.status_code} — {e.response.text}")
        return []

    data = r.json()
    local_results = data.get("local_results", [])[:5]
    activities: list[GetActivitiesResponse] = []
    
    if not local_results:
        return activities

    # 1. Preparar nombres para el LLM
    names = [item.get("title", "") for item in local_results]

    # 2. Estimación de precios con LLM
    estimator = model.with_structured_output(ActivityPrices)
    system_prompt = "Estima el precio de entrada estándar por adulto en MXN para las siguientes atracciones turísticas. Si es una plaza, parque público, mercado o iglesia, el costo es 0. Responde solo con los datos."
    human_prompt = "Atracciones: " + ", ".join(names)
    
    price_map = {}
    try:
        estimation = estimator.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        price_map = {p.name.lower(): p.estimated_price_mxn for p in estimation.prices}
    except Exception as e:
        print(f"Error al estimar precios: {e}")

    # 3. Ensamblar la respuesta
    for item in local_results:
        title = item.get("title", "Sin nombre")
        price = price_map.get(title.lower(), 0.0)
        
        activities.append(
            GetActivitiesResponse(
                name=title,
                price_per_person=price,
                duration="N/A",
                category=item.get("type", "Atracción"),
                description=f"{item.get('address', '')} (Calificación: {item.get('rating', 'N/A')} con {item.get('reviews', 0)} reseñas)",
            )
        )

    print(f"ACTIVITIES encontradas y estimadas: {len(activities)}")
    return activities


@tool("calc_budget", description="Calcula el presupuesto total del viaje.")
def calc_budget(
    price_hotel: float | str,
    price_flight: float | str,
    price_activities: float | str,
    persons: int | str,
    nights: int | str,
) -> float:
    """
    Fórmula:
        total = (vuelo × personas) + (hotel × noches) + (actividades × personas)

    Nota: price_activities debe ser el costo total de actividades POR persona.
    """
    p = int(persons)
    n = int(nights)
    flight = float(price_flight)
    hotel = float(price_hotel)
    activities = float(price_activities)

    return (flight * p) + (hotel * n) + (activities * p)