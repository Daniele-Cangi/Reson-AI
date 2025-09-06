import os, glob, json
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
from peft import PeftModel, prepare_model_for_kbit_training
from huggingface_hub import login

def setup_new_training_folder(training_dir):
    os.makedirs(training_dir, exist_ok=True)
    jsonl_files = glob.glob(os.path.join(training_dir, "*.jsonl"))
    return len(jsonl_files) > 0, jsonl_files

def continue_training_new_folder(old_model_path, new_output_path, new_data_dir,
                                 base_model_id="meta-llama/Llama-2-7b-chat-hf"):
    ok, jsonl_files = setup_new_training_folder(new_data_dir)
    if not ok: return False

    hf_token = os.getenv("HUGGINGFACE_TOKEN") or input("Enter your HF Token: ").strip()
    login(token=hf_token)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token
    )
    base_model = prepare_model_for_kbit_training(base_model)

    model = PeftModel.from_pretrained(
        base_model,
        old_model_path,
        device_map="auto",
        is_trainable=True,
        token=hf_token
    )

    tokenizer = AutoTokenizer.from_pretrained(old_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for name, param in model.named_parameters():
        param.requires_grad = 'lora' in name

    new_examples = []
    for file_path in jsonl_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if 'instruction' in entry and 'response' in entry:
                        new_examples.append(entry)
                except: continue
    if not new_examples: return False

    eos = tokenizer.eos_token or "</s>"
    def preprocess_function(examples):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        for instruction, response in zip(examples["instruction"], examples["response"]):
            text = f"[INST] {instruction} [/INST] {response}{eos}"
            tok = tokenizer(text, truncation=True, padding=False, max_length=2048)
            model_inputs["input_ids"].append(tok["input_ids"])
            model_inputs["attention_mask"].append(tok["attention_mask"])
            model_inputs["labels"].append(tok["input_ids"].copy())
        return model_inputs

    dataset = Dataset.from_list(new_examples)
    train_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=os.path.join(new_output_path, "training_logs"),
        num_train_epochs=2.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=50,
        eval_strategy="no",
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    os.makedirs(new_output_path, exist_ok=True)
    trainer.save_model(new_output_path)
    return True

if __name__ == "__main__":
    print("Usage: continue_training_new_folder(old_model_path, new_output_path, new_data_dir)")
