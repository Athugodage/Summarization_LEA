from transformers import  AutoModelForSeq2SeqLM, AutoTokenizer, LEDForConditionalGeneration
import torch
from summa.pagerank_weighted import pagerank_weighted_scipy as pagerank
from nltk import sent_tokenize
from summa.commons import build_graph, remove_unreachable_nodes
from summa.preprocessing.textcleaner import init_textcleanner, filter_words, merge_syntactic_units
from summa.summarizer import _set_graph_edge_weights, _add_scores_to_sentences, _extract_most_important_sentences


def real_clean_text_by_sentences(text, language="english", additional_stopwords=None):
    init_textcleanner(language, additional_stopwords)
    original_sentences = sent_tokenize(text)  # split_sentences(text)
    filtered_sentences = original_sentences  # filter_words(original_sentences)

    return merge_syntactic_units(original_sentences, filtered_sentences)


def rank_text(
        text,
        ratio=1.0,
        words=None,
        split=False,
):
    if not isinstance(text, str):
        raise ValueError("Text parameter must be a Unicode object (str)!")

    sentences = real_clean_text_by_sentences(text)

    return _rank_text_(sentences, ratio, words, split)


def _rank_text_(
        sentences,
        ratio=1.0,
        words=None,
        split=False,
):
    graph = build_graph([sentence.token for sentence in sentences])
    _set_graph_edge_weights(graph)

    remove_unreachable_nodes(graph)

    if len(graph.nodes()) == 0:
        return [] if split else ""

    pagerank_scores = pagerank(graph)

    _add_scores_to_sentences(sentences, pagerank_scores)

    extracted_sentences = _extract_most_important_sentences(sentences, ratio, words)

    extracted_sentences.sort(key=lambda s: s.index)

    return list(map(lambda sentence: sentence.score, extracted_sentences))


def mask_rank_text(
        text,
        tokenizer,
        ratio=1.0,
        mask_ratio=0.5,
        mask_token="<mask>",
        words=None,
        split=False,
):
    sentences = real_clean_text_by_sentences(text)
    indices_scores = sorted(list(enumerate(_rank_text_(sentences, ratio, words, split))), key=lambda x: -x[1])
    sentences = list(map(lambda sentence: sentence.text, sentences))

    number_mask_sentences = 1
    masked_indices = sorted(list(map(lambda x: x[0], indices_scores[:number_mask_sentences])))
    for index in masked_indices:
        sentences[index] = "".join(
            [mask_token] * len(tokenizer(sentences[index], add_special_tokens=False)['input_ids']))

    masked_text = " ".join(sentences)

    return masked_text


def mask_rank_texts(
        texts,
        tokenizer,
        ratio=1.0,
        mask_ratio=0.5,
        mask_token="<mask>",
        words=None,
        split=False,
):
    return [mask_rank_text(text, tokenizer, ratio, mask_ratio, mask_token, words, split) for text in texts]


class NewTransformer():
    def __init__(self,
                 model_type='pegasus',
                 model_checkpoint='google/pegasus-xsum'):
        self.model_type = model_type
        self.model_checkpoint = model_checkpoint


    def summarize(self, set_of_text, max_target_length=64,
                  attention_mode='sliding_chunks'):

        self.max_target_length = max_target_length

        if self.model_type == 'pegasus':
            self.model_checkpoint = 'google/pegasus-xsum'
            return self.use_pegasus(set_of_text)

        elif self.model_type == 'longformer':
            self.model_checkpoint = 'hyesunyun/update-summarization-bart-large-longformer'
            return self.use_longformer(set_of_text,
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
