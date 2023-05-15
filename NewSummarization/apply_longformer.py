from transformers import AutoTokenizer, LEDForConditionalGeneration
import torch

MODEL_CKPT = 'hyesunyun/update-summarization-bart-large-longformer'
model = LEDForConditionalGeneration.from_pretrained(MODEL_CKPT)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

def summarize(text, model=model,
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

text = '''
And this meant that they had to be viewed in a different way. These long American shows – think Grey’s Anatomy or NCIS or even 24 – had a habit of rattling on for so long that quality control became much less important than simply keeping the train on the rails. To think of a show like this is to think of a show that burst out of the gates with a ton of promise, then dwindled on to fewer plaudits and smaller audiences until the network did the decent thing and cancelled them. It is roughly the television equivalent of wanting to split up with someone but, rather than simply dumping them, just deciding to grow more neglectfully distant until they can’t take it and break up with you.

However, a recent spate of shows demonstrate that this might no longer be the case. Two shows are ending this month on completely their terms. Bill Hader’s Barry has just two episodes left and, if this last season is any indication, promises to go out on an unapologetically bleak final note.

And then there’s Succession. Judging by interviews with Jesse Armstrong, Succession is less building to an enraptured climax and more just stopping. Armstrong has admitted that the show could very happily keep churning out episodes for years like, say, Billions. And the ending is apparently so abrupt that Sarah Snook didn’t even have an inkling that things were coming to a close until Armstrong happened to mention it during the last ever table read. Obviously this is Succession, so it’s bound to have a satisfying and complex ending, but Armstrong’s decision to end things on his own terms is admirable. Would we be happier if Succession season five was already scheduled for November 2024? Possibly. But would our enjoyment of the show decrease with every new realisation that Armstrong was just content to shuffle the pieces around a chessboard until people stopped watching? Absolutely.


'''

print(summarize(text))