from transformers import AutoTokenizer, AutoModelForCausalLM


PAUSE_TOKEN = "<pause>"
seed = 42
model_name = "BEE-spoke-data/smol_llama-220M-GQA"

# Modify tokenizer to include the <pause> token
tokenizer = AutoTokenizer.from_pretrained("/home/vincent/Documents/learned_pause/pretraining/modified_tokenizer")
# tokenizer.add_special_tokens({"additional_special_tokens": [PAUSE_TOKEN]})

model = AutoModelForCausalLM.from_pretrained("//home/vincent/Documents/learned_pause/pretraining/results/checkpoint-3000", 
                                             device_map="auto"
                                             )



prompt = "Textbook on getting Hoes\n"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids.to(model.device)
output = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=False))