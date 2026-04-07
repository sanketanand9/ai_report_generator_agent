from ollama import Client
import json

client = Client()

# Step 1: Define tools (functions)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# Step 2: Send request
response = client.chat(
    model="qwen2.5:latest",
    messages=[
        {"role": "user", "content": "What's the weather in Bangalore?"}
    ],
    tools=tools
)

# Step 3: Check if model wants to call tool
msg = response['message']

if 'tool_calls' in msg:
    tool_call = msg['tool_calls'][0]
    
    function_name = tool_call['function']['name']
    arguments = tool_call['function']['arguments']

    print("Tool to call:", function_name)
    print("Arguments:", arguments)

    # Step 4: Execute tool (your logic)
    def get_weather(city):
        return f"Weather in {city}: 28°C, Sunny"

    result = get_weather(**arguments)

    # Step 5: Send tool result back to model
    final_response = client.chat(
        model="qwen2.5:latest",
        messages=[
            {"role": "user", "content": "What's the weather in Bangalore?"},
            msg,
            {
                "role": "tool",
                "name": function_name,
                "content": result
            }
        ]
    )

    print("\nFinal Answer:")
    print(final_response['message']['content'])

else:
    print(msg['content'])