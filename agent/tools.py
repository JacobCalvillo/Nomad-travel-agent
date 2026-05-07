import os
import httpx
from dotenv import load_dotenv
load_dotenv()

import json


from datetime import date
import airportsdata

from datetime import datetime, date
from pydantic import BaseModel, Field

from langchain.tools import tool


class SearchFlightsResponse(BaseModel):
    """Output for search_flight queries"""
    airline: str = Field(description='Name of the airline')
    price: float = Field(default=0.0, description='price of flight')
    duration: str = Field(description='duration of flight')
    departure_time: str = Field(description='Time of departure of the flight')
    arrival_time: str = Field(description='Time of arrival of the flight')

class GetActivitiesResponse(BaseModel):
    name: str = Field(description='Name of the activity')
    price_per_person: float = Field(description='Price of the activity')
    description: str = Field(description='Description of the activity')
    category: str = Field(description='Category of the activity')
    duration: str = Field(description='Duration of the activity')

class SearchHotelsResponse(BaseModel):
    name: str = Field(description='Name of the hotel')
    price_per_night: float = Field(description='Price per night of the hotel.')
    stars: float = Field(description='Calification of the Hotel in scale of 1 to 5')
    amenities: list[str] = Field(description='Amenities and commodities of the hotel')

class BudgetInput(BaseModel):
    price_hotel: float
    price_flight: float
    price_activities: float
    persons: int
    nights: int

airports = airportsdata.load('IATA')

class AirportCodeResponse(BaseModel):
    city: str
    airport_code: str
    airport_name: str

@tool('get_airport_code', description='Convert a city name into an IATA airport code')
def get_airport_code(city: str) -> AirportCodeResponse:

    city = city.lower()

    for code, airport in airports.items():
        airport_city = airport.get('city', "").lower()

        if airport_city == city:
            return AirportCodeResponse(
                city=airport["city"],
                airport_code=code,
                airport_name=airport['name']
            )
    
    raise ValueError(f'No airport found for city: {city}')

@tool('search_flights', description='Performs search of flights in multiple sites. Use it when you need to know fligths')
def search_flights(passengers:int | str, origin:str, destination:str, arrival_date: date, leave_date: date | None, type_of_flight:str = 'Redondo') -> list[SearchFlightsResponse]:

    passengers = int(passengers)

    if isinstance(arrival_date, str):
        arrival_date = date.fromisoformat(arrival_date)

    if leave_date and isinstance(leave_date, str):
        leave_date = date.fromisoformat(leave_date)
    
    match type_of_flight:
        case 'Redondo':
            type_of_flight = '1'
        case 'Sencillo':
            type_of_flight = '2'
        case 'Paradas':
            type_of_flight = '3'
        case _:
            type_of_flight = '1'
    
    print("TOOL INPUT:", origin, destination)
    
    departure = get_airport_code.invoke({
        "city": origin
    })

    arrival = get_airport_code.invoke({
        "city": destination
    })

    departure_code = departure.airport_code
    arrival_code = arrival.airport_code

    params = {
        "engine": "google_flights",
        "departure_id": departure_code,
        "arrival_id": arrival_code,
        "currency": 'MXN',
        "type": type_of_flight,
        "outbound_date": arrival_date.isoformat(),
        "api_key": os.getenv('SER_API_API_KEY'),
        "adults": passengers
    }
    
    if leave_date:
        params['return_date'] = leave_date.isoformat()
    
    print("PARAMS:", params)

    try:
        r = httpx.get(
            'https://serpapi.com/search',
            params=params
        )

        r.raise_for_status()

    except httpx.HTTPStatusError as e:
        print(e.response.text)
        raise

    response = r.json()
    
    if "best_flights" not in response:
        print("SERPAPI RESPONSE:", response)

        raise ValueError(
            "No flights found"
        )

    flights: list[SearchFlightsResponse] = []

    for flight_option in response.get("best_flights", []):
        first_flight = flight_option["flights"][0]

        flights.append(
            SearchFlightsResponse(
                airline=first_flight["airline"],
                price=float(flight_option["price"]),
                duration=f"{flight_option['total_duration']} min",
                departure_time=first_flight["departure_airport"]["time"],
                arrival_time=first_flight["arrival_airport"]["time"]
            )
        )
    print("FLIGHTS", flights)
    return flights

@tool('search_hotels', description='Retrieve information of multiple hotels. Use this when you need to perform a search of multiple hotels')
def search_hotels(place:str, check_in_date: str, check_out_date:str, adults: int, children:int) -> list[SearchHotelsResponse]:
    params = {
        "engine": "google_hotels",
        "q": f"Hotels+in+{place}",
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": adults,
        "children": children,
        "currency": "MXN",
        "api_key": os.getenv("SER_API_API_KEY")
    }
    
    if children > 0:
        params["children_ages"] = ",".join(
            ["5"] * children
        )
    
    r = httpx.get('https://serpapi.com/search',params=params)
    
    response = r.json()
    
    hotels: list[SearchHotelsResponse] = []
    
    for hotel in response.get('properties', [])[:5]:
        
        price_str = hotel.get('rate_per_night', {}).get('lowest', '0')
        price = float(price_str.replace('MX$', '').replace(',', '').strip())
        
        hotels.append(
            SearchHotelsResponse(
                name=hotel.get('name'),
                price_per_night=price,
                stars=hotel.get('overall_rating'),
                amenities=hotel.get('amenities', [])
            )
        )
    
    return hotels

@tool('get_activities', description='Retrieve information of activities near the specified zone or place. Use when user want to know what to do in this place.')
def get_activities(place: str) -> list[GetActivitiesResponse]:
    return [
        {"name": "Tour Monte Albán", "price_per_person": 350.0, "duration": "4h", "category": "cultural", 'description': 'some description.'},
        {"name": "Clase de cocina oaxaqueña", "price_per_person": 600.0, "duration": "3h", "category": "gastronomía", 'description': 'some description.'},
        {"name": "Mercado 20 de Noviembre", "price_per_person": 0.0, "duration": "2h", "category": "free", 'description': 'some description.'},
    ]

@tool('calc_budget',description="Compute total travel budget")
def calc_budget(
    price_hotel: float | str,
    price_flight: float | str,
    price_activities: float | str,
    persons: int | str,
    nights: int | str
) -> float:

    return (
        (float(price_activities) * int(persons))
        + (float(price_flight) * int(persons))
        + (float(price_hotel) * int(nights))
    )

