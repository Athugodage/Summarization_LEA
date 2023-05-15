from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, LEDForConditionalGeneration
import torch

class NewTransformer():
    def __init__(self,
                 model_type='pegasus',
                 model_checkpoint='google/pegasus-xsum'):
        self.model_type = model_type
        self.model_checkpoint = model_checkpoint

    def summarize(self, text, max_target_length=64,
                  attention_mode='sliding_chunks'):
        if self.model_type == 'pegasus':
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            use_pegasus(text,
                        model=model,
                        tokenizer=tokenizer,
                        max_target_length=max_target_length)

        elif self.model_type == 'longformer':
            model = LEDForConditionalGeneration.from_pretrained(self.model_checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            return use_longformer(text,
                                  model=model,
                                  tokenizer=tokenizer,
                                  max_target_length=max_target_length,
                                  attention_mode='sliding_chunks')



def  use_pegasus(text, model_checkpoint='google/pegasus-xsum'):

    MODEL_CKPT = 'google/pegasus-xsum'
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CKPT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

    inputs_dict = tokenizer(text, padding=True, return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids
    attention_mask = inputs_dict.attention_mask

    predicted_summary_ids = model.generate(input_ids, attention_mask=attention_mask)

    return tokenizer.batch_decode(predicted_summary_ids,
                                  max_length=max_target_length, skip_special_tokens=True)


def use_longformer(text,
                   model=model,
                   tokenizer=tokenizer,
                   max_target_length=64,
                   attention_mode='sliding_chunks'):


    config = model.config
    # choose the attention mode 'n2', 'tvm' or 'sliding_chunks'
    # 'n2': for regular n2 attention
    # 'tvm': a custom CUDA kernel implementation of our sliding window attention
    # 'sliding_chunks': a PyTorch implementation of our sliding window attention
    config.attention_mode = attention_mode


    inputs_dict = tokenizer(text, padding=True,  return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids
    attention_mask = inputs_dict.attention_mask
    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1


    predicted_summary_ids = model.generate(input_ids,
                                           attention_mask=attention_mask,
                                           global_attention_mask=global_attention_mask)
    return tokenizer.batch_decode(predicted_summary_ids,
                                  max_length=max_target_length, skip_special_tokens=True)
