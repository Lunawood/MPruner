#!/usr/bin/env python
# coding: utf-8


import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_embeddings import SentenceEmbeddings

class QuestionGeneration:

    def __init__(self, model_path=None,tokenizer_path=None,use_hg=False):
        
        if use_hg:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        else:
            self.model = torch.load(model_path)

        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, answer: str, context: str):
        input_text = '<answer> %s <context> %s ' % (answer, context)
        encoding = self.tokenizer.encode_plus(
            input_text,
            return_tensors='pt'
        ).to(self.device)
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams = 3,
            num_return_sequences = 1
        )
        question_list = []
        for output in outputs:
            question = self.tokenizer.decode(
                output,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            question_list.append({'question': question, 'answer': answer, 'context': context})
        return question_list



