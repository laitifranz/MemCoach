import requests
from dotenv import load_dotenv
import os

load_dotenv()


class OpenRouterUtils:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")

    def get_credits(self):
        url = "https://openrouter.ai/api/v1/credits"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers)
        print(response.json()["data"])

    def compute_total_api_cost(
        self,
        input_tokens,
        output_tokens,
        price_per_1M_input,
        price_per_1M_output,
        num_api_calls,
    ):
        price_per_input_token = price_per_1M_input / 1_000_000
        price_per_output_token = price_per_1M_output / 1_000_000

        # Compute cost per call
        cost_per_call = (input_tokens * price_per_input_token) + (
            output_tokens * price_per_output_token
        )

        # Total cost for all calls
        total_cost = cost_per_call * num_api_calls

        print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    """
    Example usage:
    uv run src/utils/_openrouter_utils.py compute_total_api_cost --input_tokens=700 --output_tokens=200 --price_per_1M_input=0.25 --price_per_1M_output=2 --num_api_calls=80000
    uv run src/utils/_openrouter_utils.py get_credits
    """
    import fire

    fire.Fire(OpenRouterUtils)
