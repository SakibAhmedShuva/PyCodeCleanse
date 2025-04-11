!pip install bitsandbytes unsloth bitsandbytes msgspec blake3 fastapi gguf
!pip install gguf
from unsloth import FastModel
import torch
fourbit_models = [
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
] 
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it",
    max_seq_length = 2048, 
    load_in_4bit = True,  
    load_in_8bit = False, 
    full_finetuning = False, 
)
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, 
    finetune_language_layers   = True,  
    finetune_attention_modules = True,  
    finetune_mlp_modules       = True,  
    r = 8,           
    lora_alpha = 8,  
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)
from datasets import load_dataset
dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)
dataset[100]
def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return { "text" : texts }
pass
dataset = dataset.map(apply_chat_template, batched = True)
dataset[100]["text"]
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, 
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, 
        warmup_steps = 5,
        max_steps = 30,
        learning_rate = 2e-4, 
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", 
    ),
)
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)
tokenizer.decode(trainer.train_dataset[100]["input_ids"])
tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
trainer_stats = trainer.train()
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)
messages = [{
    "role": "user",
    "content": [{
        "type" : "text",
        "text" : "Continue the sequence: 1, 1, 2, 3, 5, 8,",
    }]
}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, 
)
outputs = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 64, 
    temperature = 1.0, top_p = 0.95, top_k = 64,
)
tokenizer.batch_decode(outputs)
messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Why is the sky blue?",}]
}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, 
)
from transformers import TextStreamer
_ = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 64, 
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)
model.save_pretrained("gemma-3")  
tokenizer.save_pretrained("gemma-3")
if False:
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name = "lora_model", 
        max_seq_length = 2048,
        load_in_4bit = True,
    )
messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "What is Gemma-3?",}]
}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, 
)
from transformers import TextStreamer
_ = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 64, 
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)
if False: 
    model.save_pretrained_merged("gemma-3-finetune", tokenizer)
if False: 
    model.push_to_hub_merged(
        "HF_ACCOUNT/gemma-3-finetune", tokenizer,
        token = "hf_..."
    )
if False: 
    model.save_pretrained_gguf(
        "gemma-3-finetune",
        quantization_type = "Q8_0", 
    )
if False: 
    model.push_to_hub_gguf(
        "gemma-3-finetune",
        quantization_type = "Q8_0", 
        repo_id = "HF_ACCOUNT/gemma-finetune-gguf",
        token = "hf_...",
    )