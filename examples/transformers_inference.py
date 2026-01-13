"""
Example: Using the optimized configuration for inference.

This example shows how to apply an optimized configuration
from KVCache Auto-Tuner to your inference pipeline.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def load_optimized_model(model_id: str):
    """
    Load model with optimized settings from KVCache Auto-Tuner.

    This configuration was determined by running:
        kvat tune <model_id> --profile chat-agent
    """
    # Optimized settings (example output from kvat)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Optimized dtype
        attn_implementation="sdpa",  # Optimized attention
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Setup optimized cache
    cache = DynamicCache()

    return model, tokenizer, cache


def generate_optimized(
    model,
    tokenizer,
    cache,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate text using optimized configuration."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            past_key_values=cache,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def main():
    """Run inference example."""
    model_id = "meta-llama/Llama-3.2-1B"

    print(f"Loading optimized model: {model_id}")
    model, tokenizer, cache = load_optimized_model(model_id)

    # Example prompts
    prompts = [
        "Explain the concept of KV-cache in transformer models:",
        "What are the benefits of using flash attention?",
        "How does dynamic caching differ from static caching?",
    ]

    for prompt in prompts:
        print(f"\n{'=' * 50}")
        print(f"Prompt: {prompt}")
        print(f"{'=' * 50}")

        response = generate_optimized(
            model,
            tokenizer,
            cache,
            prompt,
            max_new_tokens=256,
        )

        print(f"\nResponse:\n{response}")

        # Reset cache between prompts for independent generations
        if hasattr(cache, "reset"):
            cache.reset()


if __name__ == "__main__":
    main()
