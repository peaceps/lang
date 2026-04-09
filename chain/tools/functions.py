from langchain_core.tools import tool
from urllib.error import HTTPError
from pydantic import BaseModel, Field
import requests
import datetime
from datetime import timedelta
import wikipedia


class TemperatureInput(BaseModel):
    """The location to point the temperature."""
    latitude: float = Field(description="The latitude of the location.")
    longitude: float = Field(description="The longitude of the location.")


@tool(args_schema=TemperatureInput)
def get_temperature(latitude: float, longitude: float) -> str:
    """Fetch the temperature at the given location."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m",
        "forecast_days": 1,
    }
    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        results = response.json()
    else:
        raise HTTPError(response.url, response.status_code, f"Failed to fetch temperature: {response.status_code}")

    current_utc_time = datetime.datetime.now(datetime.UTC)
    time_list = [datetime.datetime.fromisoformat((current_utc_time + timedelta(hours=i)).isoformat()) for i in range(24)]
    temperature_list = results["hourly"]["temperature_2m"]
    closest_temperature_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_temperature_index]
    return f'Current temperature at {latitude}, {longitude} is {current_temperature}°C'


@tool
def search_wikipedia(query: str) -> str:
    """Search the Wikipedia for the given query and get the summary."""
    titles = wikipedia.search(query)
    summaries = []
    for page_title in titles[:3]:
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except wikipedia.exceptions.PageError:
            pass
    if not summaries:
        return "No summary found for the given query."
    return "\n\n".join(summaries)
