# Assume Groq Llama-3-8B costs roughly $0.05 per 1M tokens
COST_PER_INPUT_TOKEN = 0.00000005
COST_PER_OUTPUT_TOKEN = 0.00000008

def calculate_run_cost(total_input_tokens: int, total_output_tokens: int) -> dict:
    input_cost = total_input_tokens * COST_PER_INPUT_TOKEN
    output_cost = total_output_tokens * COST_PER_OUTPUT_TOKEN
    total_cost = input_cost + output_cost
    
    return {
        "input_cost_usd": round(input_cost, 4),
        "output_cost_usd": round(output_cost, 4),
        "total_cost_usd": round(total_cost, 4)
    }