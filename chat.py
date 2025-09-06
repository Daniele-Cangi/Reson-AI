#!/usr/bin/env python3
"""
RESON-LLAMA Chat con MEMORIA CONVERSAZIONALE - PULIZIA MINIMALE
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import warnings
import re

warnings.filterwarnings("ignore", category=UserWarning)

conversation_turns = []
MAX_MEMORY_TURNS = 4

def load_reson_model(model_path=r"C:\Users\dacan\OneDrive\Desktop\Meta\Reson4.5\Reson4.5"):
    print(f"🧠 Caricamento RESON-LLAMA da {model_path}...")
    
    base_model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
        low_cpu_mem_usage=True
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    
    print("✅ RESON-LLAMA V4 caricato con memoria!")
    return model, tokenizer

def minimal_clean_response(response):
    """Pulizia MINIMALE - rimuove tutto tra parentesi quadre"""
    
    # Rimuovi QUALSIASI cosa tra parentesi quadre [...]
    cleaned = re.sub(r'\[.*?\]', '', response)
    
    # Pulizia spazi multipli
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r' *\n *', '\n', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

def format_conversation_prompt(conversation_turns, current_question):
    prompt_parts = []
    
    for turn in conversation_turns[-MAX_MEMORY_TURNS:]:
        prompt_parts.append(f"[INST] {turn['question']} [/INST] {turn['answer']}")
    
    prompt_parts.append(f"[INST] {current_question} [/INST]")
    
    full_prompt = " ".join(prompt_parts)
    return full_prompt

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    input_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.60,
            do_sample=True,
            top_p=0.94,
            top_k=40,
            min_p=0.05,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            min_length=60,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    new_tokens = outputs[0][input_length:]
    raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    
    # Pulizia minimale - mantieni tutto il contenuto interessante
    clean_response = minimal_clean_response(raw_response)
    
    return clean_response

def chat_with_memory(model, tokenizer):
    global conversation_turns
    conversation_turns = []
    
    print("\n🧠 RESON-LLAMA V4 CHAT CON MEMORIA")
    print("Comandi: 'quit' = esci, 'clear' = cancella memoria")
    
    while True:
        try:
            user_input = input(f"\n🧑 Tu: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Arrivederci!")
                break
                
            elif user_input.lower() == 'clear':
                conversation_turns = []
                print("🧠 Memoria cancellata!")
                continue
            
            if not user_input:
                continue
            
            print("🧠 RESON sta riflettendo...")
            
            prompt = format_conversation_prompt(conversation_turns, user_input)
            response = generate_response(model, tokenizer, prompt)
            
            print(f"\n🤖 RESON: {response}")
            
            conversation_turns.append({
                'question': user_input,
                'answer': response
            })
            
            if len(conversation_turns) > MAX_MEMORY_TURNS:
                conversation_turns = conversation_turns[-MAX_MEMORY_TURNS:]
            
        except KeyboardInterrupt:
            print("\n👋 Chat interrotta!")
            break
        except Exception as e:
            print(f"❌ Errore: {e}")

def main():
    print("🧠 RESON-LLAMA V4 CON MEMORIA")
    
    model, tokenizer = load_reson_model()
    chat_with_memory(model, tokenizer)

if __name__ == "__main__":
    main()