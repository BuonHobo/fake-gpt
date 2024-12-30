prompt = """\
What is 13-3?
<request><SimpleCalculatorTool>13-3<call>10.0<response>
Result=10<submit>
"""

def reward_fn(result, answer):
    """Simplified reward function returning 1 if result matches answer and 0 otherwise."""
    result_parsed = result.split("=")[1].split("<")[0]
    return int(result_parsed==answer)

text_env = TextEnvironemnt(
    model=model, 
    tokenizer=tokenizer,
    tools= {"SimpleCalculatorTool": load_tool("ybelkada/simple-calculator")},
    reward_fn=exact_match_reward,
    prompt=prompt, 
    max_turns=1,
    max_tool_response=100,
    generation_kwargs={"do_sample": "true"},
)