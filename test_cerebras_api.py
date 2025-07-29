# Import necessary libraries
import os
import dotenv

# Import the Cerebras SDK
from cerebras.cloud.sdk import Cerebras

# Load environment variables from .env file
dotenv.load_dotenv()

# Create a Cerebras client with API key
client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

# List available models
print(client.models.list())

# Retrieve a specific model (qwen-3-32b)
print(client.models.retrieve("qwen-3-32b"))
print(client.models.retrieve("qwen-3-235b-a22b"))
print(client.models.retrieve("qwen-3-235b-a22b-instruct-2507"))