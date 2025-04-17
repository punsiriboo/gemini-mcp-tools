from typing import List
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import asyncio
from dotenv import load_dotenv

# STEP 1: Configure API key
load_dotenv(override=True)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("❌ GEMINI_API_KEY is not set in the environment variables.")

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["mcp_server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


async def agent_loop(prompt: str, client: genai.Client, session: ClientSession):
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    await session.initialize()
    
    # --- 1. Get Tools from Session and convert to Gemini Tool objects ---
    # --- [MCP] MCP Client Session - List Tool for Gemini
    mcp_tools = await session.list_tools()
    tools = types.Tool(function_declarations=[
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        }
        for tool in mcp_tools.tools
    ])
    
    # --- 2. Initial Request ---
    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0,
            tools=[tools],
        ),
    )
    
    contents.append(response.candidates[0].content)

    # --- 3. Tool Calling Loop ---
    turn_count = 0
    max_tool_turns = 5
    while response.function_calls and turn_count < max_tool_turns:
        turn_count += 1
        tool_response_parts: List[types.Part] = []

        for fc_part in response.function_calls:
            tool_name = fc_part.name
            args = fc_part.args or {}
            print(f"Attempting to call MCP tool: '{tool_name}' with args: {args}")

            try:
                # --- [MCP] MCP Client Session - call too from MCP Local Server
                tool_result = await session.call_tool(tool_name, args)
                if tool_result.isError:
                    tool_response = {"error": tool_result.content[0].text}
                else:
                    tool_response = {"result": tool_result.content[0].text}
            except Exception as e:
                tool_response = {"error": f"Tool execution failed: {type(e).__name__}: {e}"}
            
            print(f"Tool response: {tool_response}")
            tool_response_parts.append(
                types.Part.from_function_response(
                    name=tool_name, response=tool_response
                )
            )
        # print(f"Tool response parts: {tool_response}")
        contents.append(types.Content(role="user", parts=tool_response_parts))
        print(f"Added {len(tool_response_parts)} tool response parts to history.")

        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=1.0,
                tools=[tools],
            ),
        )
        contents.append(response.candidates[0].content)

    if turn_count >= max_tool_turns and response.function_calls:
        print(f"Maximum tool turns ({max_tool_turns}) reached. Exiting loop.")

    print("MCP tool calling loop finished. Returning final response.")
    return response

async def run():
    prompt = input("💬 Please enter your prompt: ").strip()
    if not prompt:
        print("⚠️ No prompt provided. Exiting.")
        return

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            print(f"Running agent loop with prompt: {prompt}")
            res = await agent_loop(prompt, client, session)
            print("🧠 AI Response:")
            print(res.text)

if __name__ == "__main__":
    asyncio.run(run())