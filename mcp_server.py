from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("My Currecncy Converter MCP")

@mcp.tool()
def exchange_rate_convert(base: str, target: str, amount: float) -> str:
    """Convert currency from base to target with amount"""
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

        return f"{amount} {base} = {converted_amount:.2f} {target} (Rate: 1 {base} = {rate} {target}, as of {date})"
    except requests.exceptions.RequestException as e:
        return f"Failed to fetch exchange rate: {e}"

if __name__ == "__main__":
    mcp.run(transport='stdio')