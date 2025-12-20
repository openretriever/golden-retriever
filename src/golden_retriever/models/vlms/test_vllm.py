from vllm import LLM

prompts = ["Hello, my name is", "The capital of France is"]  # Sample prompts.
# llm = LLM(model="lmsys/vicuna-7b-v1.3")  # Create an LLM.
llm = LLM(
    model="allenai/Molmo-7B-D-0924",
    model_type="llama",  # FXIME Treat it as a Llama model
    trust_remote_code=True
)
outputs = llm.generate(prompts)  # Generate texts from the prompts.

print("outputs:")
print(outputs)