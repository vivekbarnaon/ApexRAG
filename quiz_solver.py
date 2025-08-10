from __future__ import annotations

from typing import Dict, Optional

import requests
import traceback

from landmark_vector_db import get_landmark_for_city


def solve_quiz(city_api_url: str, flight_api_urls: Dict[str, str], client) -> Optional[str]:
    """Get city from API, find its landmark, then get flight number from corresponding API."""
    # Step 1: Get the favorite city
    try:
        city_response = requests.get(city_api_url, timeout=10)
        city_response.raise_for_status()
        favorite_city_data = city_response.json()
        
        favorite_city = (favorite_city_data or {}).get("data", {}).get("city")
        
        if not favorite_city:
            return None
    except requests.exceptions.RequestException as req_e:
        return None
    except Exception as e:
        return None

    # Step 2: Map city to landmark using FAISS vector DB
    associated_landmark = get_landmark_for_city(client, favorite_city)
    
    if not associated_landmark:
        return None

    # Step 3: Choose the correct flight path API
    if associated_landmark == "Gateway of India":
        flight_api_url = flight_api_urls.get("gateway_of_india")
    elif associated_landmark == "Taj Mahal":
        flight_api_url = flight_api_urls.get("taj_mahal")
    elif associated_landmark == "Eiffel Tower":
        flight_api_url = flight_api_urls.get("eiffel_tower")
    elif associated_landmark == "Big Ben":
        flight_api_url = flight_api_urls.get("big_ben")
    else:
        flight_api_url = flight_api_urls.get("other_landmarks")

    if not flight_api_url:
        return None

    # Step 4: Fetch flight number
    try:
        flight_response = requests.get(flight_api_url, timeout=10)
        flight_response.raise_for_status()
        flight_number_data = flight_response.json()
        
        flight_number = (flight_number_data or {}).get("data", {}).get("flightNumber")
        
        if not flight_number:
            return None
        
        return str(flight_number)
        
    except requests.exceptions.RequestException as req_e:
        return None
    except Exception as e:
        return None


