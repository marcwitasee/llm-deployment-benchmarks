from dotenv import load_dotenv
import os
load_dotenv()
print("ANTHROPIC_API_KEY" in os.environ, "FIREWORKS_API_KEY" in os.environ)