from ray import serve
from transformers import pipeline
import requests

serve.start()

@serve.deployment
def model(request):
    language_model = pipeline("text-generation", model="gpt-2")
    query = request.query_params["query"]
    return language_model(query, max_length=100)

model.deploy()

query = "What is the meaning of life?"
response = requests.get(f"http://localhost:8000/model?query={query}")
print(response.text)