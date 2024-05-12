import os
from peft import AdaLoraConfig, get_peft_model
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import datasets
import transformers

dataset = datasets.load_from_disk("./data/hg/banks_sentenized_w_emp")

base_model_id = "/external/nfs/lamorozov/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/b70aa86578567ba3301b21c8a27bea4e8f6d6d61"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token


def formatting_func(example):
    """Gen. input text based on a prompt, task instruction, (context info.), and answer
  
    :param data_point: dict: Data point
    :return: dict: tokenzed prompt
    """
    prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
    instruction = """Ты занимаешься аспектно-ориентированным анализом настроения клиентов.
Ты выделяешь из отзывов клиентов полярности для терминов аспектов (aspect term) и терминов мнения (opinion term).
Термины аспектов (aspect terms) -- конкретного элемента или характеристика товара, продукта или сервиса, которую анализируют для определения настроения или отношения. Аспектами могут быть: качество, цена, удобство использования и т.д..
Термины мнения (opinion terms) -- выражения, отражающие отношение клиента к аспекту.
Полярность (sentiment polarity) -- отношение клиента к аспекту, выражаемое терминами мнения. Примает одно значение из "POS" или "NEG".

Твой ответ обязательно должен соответствовать формату JSON. Схема ответа:
{{
    "triplets": [
        {{
            // Характеристика
            "aspect_term": string,
            // Термины мнения
            "opinion_term": string,
            // Полярность
            "sentiment": "POS" или "NEG",
        }},
        ...
    ]
}}
"""
    from gigachain_extensions.pydantic_models import ASTEAnswer
    text = f"""<s>[INST]{prefix_text} {instruction} Вход: {example['text']} [/INST]Ответ: {ASTEAnswer(triplets=example['triplets']).model_dump_json()}</s>"""
    return text

max_length = 600 # This was an appropriate max length for my dataset

def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_train_dataset = dataset['train'].map(generate_and_tokenize_prompt2)
tokenized_val_dataset = dataset['val'].map(generate_and_tokenize_prompt2)

for i in range(5):
    print(tokenizer.decode(tokenized_train_dataset[i]['input_ids']))
    print("-" * 10)


from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


config = AdaLoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# %%
if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

project = "instruct-finetune-json-adalora-r16"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        # max_steps=1000,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        bf16=True,
        # optim="paged_adamw_8bit",
        logging_steps=100,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="epoch",       # Save the model checkpoint every logging step
        # save_steps=25,                # Save checkpoints every 50 steps
        evaluation_strategy="epoch", # Evaluate the model every logging step
        # eval_steps=25,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        num_train_epochs=2,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


