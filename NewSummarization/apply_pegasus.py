from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_CKPT = 'google/pegasus-xsum'
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CKPT)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

def summarize(text, model=model,
              tokenizer=tokenizer,
              max_target_length=64):

    inputs_dict = tokenizer(text, padding=True, return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids
    attention_mask = inputs_dict.attention_mask

    predicted_summary_ids = model.generate(input_ids, attention_mask=attention_mask)

    return tokenizer.batch_decode(predicted_summary_ids,
                                  max_length=max_target_length, skip_special_tokens=True)

