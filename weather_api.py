import requests

API_KEY = "9297baaf36f25e3a523f979be8fc6710"  # ✅ Paste your actual key here

def get_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        print(" Fetching:", url)  # Debug print
        response = requests.get(url)
        data = response.json()
        print(" Response:", data)  # Debug print

        if data.get("cod") != 200:
            print(" Error fetching weather data:", data)
            return None, None, None

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rainfall = data.get("rain", {}).get("1h", 0)  # Default 0
        return temperature, humidity, rainfall
    except Exception as e:
        print(" Exception while fetching weather:", e)
        return None, None, None