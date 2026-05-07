import os
import httpx
from dotenv import load_dotenv
load_dotenv()

from datetime import date

import airportsdata

from pydantic import BaseModel, Field
from langchain.tools import tool


# =========================================================
# Response schemas
# =========================================================

class SearchFlightsResponse(BaseModel):
    """Resultado de búsqueda de vuelos."""
    airline: str = Field(description="Nombre de la aerolínea")
    price: float = Field(default=0.0, description="Precio del vuelo en MXN")
    duration: str = Field(description="Duración total del vuelo")
    departure_time: str = Field(description="Hora de salida")
    arrival_time: str = Field(description="Hora de llegada")


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


def _get_airport_code(city: str) -> str:
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

    params = {
        "engine": "google_flights",
        "departure_id": departure_code,
        "arrival_id": arrival_code,
        "currency": "MXN",
        "type": type_code,
        "outbound_date": arrival_date.isoformat(),
        "api_key": os.getenv("SER_API_API_KEY"),
        "adults": passengers,
        "no_cache": "true",
        "gl": "mx",
        "hl": "es"
    }

    if leave_date:
        params["return_date"] = leave_date.isoformat()

    try:
        r = httpx.get("https://serpapi.com/search", params=params, timeout=15.0)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        print(f"SERPAPI ERROR: {e.response.status_code} — {e.response.text}")
        raise

    response = r.json()

    # SerpAPI a veces retorna other_flights en lugar de best_flights
    raw = response.get("best_flights") or response.get("other_flights", [])

    if not raw:
        print("SERPAPI: sin resultados →", list(response.keys()))
        return []

    flights: list[SearchFlightsResponse] = []

    for option in raw:
        first_leg = option["flights"][0]
        flights.append(
            SearchFlightsResponse(
                airline=first_leg["airline"],
                price=float(option["price"]),
                duration=f"{option['total_duration']} min",
                departure_time=first_leg["departure_airport"]["time"],
                arrival_time=first_leg["arrival_airport"]["time"],
            )
        )

    print(f"FLIGHTS encontrados: {len(flights)}")
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


@tool("get_activities", description="Obtiene actividades turísticas disponibles en el destino.")
def get_activities(place: str) -> list[GetActivitiesResponse]:
    """
    TODO: conectar a API de actividades
    """
    return [
        GetActivitiesResponse(
            name="Tour Monte Albán",
            price_per_person=350.0,
            duration="4h",
            category="cultural",
            description=f"Visita guiada a la zona arqueológica cercana a {place}.",
        ),
        GetActivitiesResponse(
            name="Clase de cocina local",
            price_per_person=600.0,
            duration="3h",
            category="gastronomía",
            description=f"Aprende a preparar platillos típicos de {place}.",
        ),
        GetActivitiesResponse(
            name="Recorrido por mercado local",
            price_per_person=0.0,
            duration="2h",
            category="free",
            description=f"Explora el mercado principal de {place} con guía.",
        ),
    ]


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