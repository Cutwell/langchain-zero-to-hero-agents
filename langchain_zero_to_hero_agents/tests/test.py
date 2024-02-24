import requests

response = requests.post(
    "http://localhost:8000/agent/invoke",
    json={'input': {"input": "what is the weather in new york"}}
)

print(response.json())