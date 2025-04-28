# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")



training_args = DPOConfig(
     output_dir="Qwen2.5-0.5B-DPO",
     logging_steps=50,
     per_device_train_batch_size=8,      # 用足一半显存
     gradient_accumulation_steps=1,      # 不用累积，step/sec 最多
     fp16=True,
     gradient_checkpointing=True,
     max_length=1024,
     max_prompt_length=256,
     save_steps = 20000,
     save_total_limit = 1,
     loss_type='DPO'
)
# we directly change the DPOTrainer local file to modify and create our own methods.
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
# accelerate launch --mixed_precision=fp16 11711-hw4/train_dpo.py