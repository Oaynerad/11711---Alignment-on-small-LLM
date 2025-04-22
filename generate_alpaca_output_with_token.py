import json
import os
import time
from vllm import LLM, SamplingParams
import datasets
from transformers import AutoTokenizer  # Add tokenizer import

def generate_alpaca_eval_outputs(model_path, batch_size=16, output_file="qwen_dpo_vllm_outputs.json"):
    print(f"Starting vLLM evaluation on AlpacaEval dataset with batch size={batch_size}")
    
    # Load evaluation dataset
    print("Loading AlpacaEval dataset...")
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    print(f"Loaded {len(eval_set)} evaluation samples")
    
    # Initialize vLLM
    print(f"Loading model: {model_path}")
    start_time = time.time()
    llm = LLM(
        model=model_path,
        dtype="float16",
        tensor_parallel_size=1,
        trust_remote_code=True
    )
    print(f"Model loading completed in {time.time() - start_time:.2f} seconds")
    
    # Load tokenizer separately to properly format prompts
    print("Loading tokenizer to get proper chat template...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set system prompt in English
    system_prompt = "You are a helpful assistant. Your responses should be helpful, accurate, harmless, and aligned with human preferences."
    
    # Prepare all prompts using the official chat template
    all_instructions = [example["instruction"] for example in eval_set]
    all_prompts = []
    
    print("Preparing prompts with proper chat template...")
    for instruction in all_instructions:
        # Create messages in the format expected by the tokenizer
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ]
        
        # Apply the model's proper chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        all_prompts.append(formatted_prompt)
    
    # Set generation parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        repetition_penalty=1.1
    )
    
    # Process in batches
    outputs = []
    total_batches = (len(all_prompts) + batch_size - 1) // batch_size
    
    print(f"Starting generation, total batches: {total_batches}")
    generation_start_time = time.time()
    
    for i in range(0, len(all_prompts), batch_size):
        batch_start_time = time.time()
        batch_prompts = all_prompts[i:i+batch_size]
        current_batch = i // batch_size + 1
        
        print(f"Processing batch {current_batch}/{total_batches}...")
        
        # Generate responses
        batch_outputs = llm.generate(batch_prompts, sampling_params)
        
        # Process generation results
        for j, output in enumerate(batch_outputs):
            idx = i + j
            if idx < len(all_instructions):
                outputs.append({
                    "instruction": all_instructions[idx],
                    "output": output.outputs[0].text.strip(),
                    "generator": "Qwen2.5-0.5B"
                })
        
        batch_time = time.time() - batch_start_time
        print(f"Batch {current_batch} completed in {batch_time:.2f} seconds")
        print(f"Progress: {min(len(outputs), len(all_instructions))}/{len(all_instructions)} ({min(len(outputs), len(all_instructions))*100/len(all_instructions):.1f}%)")
        
        # Save intermediate results every 5 batches
        if current_batch % 5 == 0 or current_batch == total_batches:
            print(f"Saving intermediate results...")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(outputs, f, ensure_ascii=False, indent=2)
    
    total_generation_time = time.time() - generation_start_time
    print(f"\nGeneration completed! Total time: {total_generation_time:.2f} seconds")
    print(f"Average time per sample: {total_generation_time/len(all_instructions):.2f} seconds")
    print(f"Results saved to {output_file}")
    
    return output_file

if __name__ == "__main__":
    # Set model path using absolute path
    model_dir = 'Qwen/Qwen2.5-0.5B-Instruct'
    
    # Run generation
    output_file = generate_alpaca_eval_outputs(model_dir, batch_size=16)
    
    print("\nAfter generation is complete, you can evaluate with these commands:")
    print("conda deactivate  # Exit vLLM environment")
    print("conda activate <original_env_name>  # Switch to AlpacaEval environment")
    print(f"export OPENAI_API_KEY=<your_openai_api_key>")
    print(f"alpaca_eval --model_outputs '{output_file}' --annotators_config 'alpaca_eval_gpt4_turbo_fn'")
    # python train-code/generate_alpaca_output_with_token.py 