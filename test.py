import requests

url = "http://192.168.50.100:11434/api/embed"
payload = {
    "model": "llama3.1:1b",
    "input": ["This is a test query."]
}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")