import os 
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# STEP 1: Configure API key
load_dotenv(override=True)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("❌ GEMINI_API_KEY is not set in the environment variables.")
client = genai.Client(api_key=api_key)


# STEP 2: Define Function (as dict)
convert_exchange_rate_function = {
    "name": "convert_exchange_rate",
    "description": "Convert any currency based the latest exchange rate and convert a specific amount between two currencies.",
    "parameters": {
        "type": "object",
        "properties": {
            "base": {
                "type": "string",
                "description": "Base currency"
            },
            "target": {
                "type": "string",
                "description": "Target currency"
            },
            "amount": {
                "type": "number",
                "description": "The amount to convert from the base currency to the target currency."
            }
        },
        "required": ["base", "target", "amount"]
    }
}

# STEP 3: Convert to Gemini FunctionDeclaration
convert_exchange_rate_declaration = types.FunctionDeclaration(
    name=convert_exchange_rate_function["name"],
    description=convert_exchange_rate_function["description"],
    parameters=convert_exchange_rate_function["parameters"]
)


# STEP 4: Define the actual function logic
def exchange_rate_convert(base: str, target: str, amount: float) -> str:
    url = f"https://api.frankfurter.dev/v1/latest?base={base}&symbols={target}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "rates" not in data or target not in data["rates"]:
            raise ValueError(f"Unexpected API response: {data}")
        rate = data["rates"][target]
        date = data["date"]
        converted_amount = amount * rate
        return f"💱 {amount} {base} = {converted_amount:.2f} {target} (Rate: 1 {base} = {rate} {target}, as of {date})"
    except requests.exceptions.RequestException as e:
        return f"Failed to fetch exchange rate: {e}"
    except ValueError as e:
        return f"Data error: {e}"


if __name__ == "__main__":
    # Chat session
    contents = []

    print("💬 Start chatting with Gemini! Type 'exit' to end the session.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("👋 Ending chat session. Goodbye!")
            break
        else:
            print("🤖 Gemini is thinking...")
            # Add user input to the conversation
            contents.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

            # STEP 5: Tool setup for Gemini
            tool = types.Tool(function_declarations=[convert_exchange_rate_declaration])
            config = types.GenerateContentConfig(tools=[tool])
            # Generate response from Gemini
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=config,
            )

            # Handle function call
            tool_call = response.candidates[0].content.parts[0].function_call

            if tool_call:
                print(f"🔧 Gemini decided to call: {tool_call.name}")
                print(f"📦 Args: {tool_call.args}")
                try:
                    # Call the real exchange rate function
                    result = exchange_rate_convert(**tool_call.args)
                    print(f"Gemini: {result}")
                    # Append the function call and result to the conversation
                    contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
                    function_response_part = types.Part.from_function_response(
                        name=tool_call.name,
                        response={"result": result},
                    )
                    contents.append(types.Content(role="user", parts=[function_response_part]))
                except Exception as e:
                    print(f"Error during function execution: {e}")
            else:
                # Print Gemini's plain response
                gemini_response = response.candidates[0].content.parts[0].text
                print(f"Gemini: {gemini_response}")
                contents.append(types.Content(role="model", parts=[types.Part(text=gemini_response)]))