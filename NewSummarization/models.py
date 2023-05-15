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

        self.max_target_length = max_target_length

        if self.model_type == 'pegasus':
            self.model_checkpoint = 'google/pegasus-xsum'
            return self.use_pegasus(text)

        elif self.model_type == 'longformer':
            self.model_checkpoint = 'hyesunyun/update-summarization-bart-large-longformer'
            return self.use_longformer(text,
                                  attention_mode=attention_mode)



    def _preprocess(self, set_of_text):
        return mask_rank_texts(set_of_text, tokenizer=self.tokenizer)

    def  use_pegasus(self, set_of_text):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        masked_dataset = self._preprocess(set_of_text)

        inputs_dict = self.tokenizer(masked_dataset, padding=True, return_tensors="pt", truncation=True)
        input_ids = inputs_dict.input_ids
        attention_mask = inputs_dict.attention_mask

        predicted_summary_ids = model.generate(input_ids, attention_mask=attention_mask)

        return self.tokenizer.batch_decode(predicted_summary_ids,
                                      max_length=self.max_target_length, skip_special_tokens=True)


    def use_longformer(self, set_of_text,
                       attention_mode='sliding_chunks'):

        model = LEDForConditionalGeneration.from_pretrained(self.model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        config = model.config
        # choose the attention mode 'n2', 'tvm' or 'sliding_chunks'
        # 'n2': for regular n2 attention
        # 'tvm': a custom CUDA kernel implementation of our sliding window attention
        # 'sliding_chunks': a PyTorch implementation of our sliding window attention
        config.attention_mode = attention_mode

        masked_dataset = self._preprocess(set_of_text)

        inputs_dict = self.tokenizer(masked_dataset, padding=True,  return_tensors="pt", truncation=True)
        input_ids = inputs_dict.input_ids
        attention_mask = inputs_dict.attention_mask
        global_attention_mask = torch.zeros_like(attention_mask)
        # put global attention on <mask> token
        global_attention_mask[:, 0] = 1


        predicted_summary_ids = model.generate(input_ids,
                                               attention_mask=attention_mask,
                                               global_attention_mask=global_attention_mask)
        return self.tokenizer.batch_decode(predicted_summary_ids,
                                      max_length=self.max_target_length, skip_special_tokens=True)
