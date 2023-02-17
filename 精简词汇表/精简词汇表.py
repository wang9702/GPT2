from transformers import BertModel, BertConfig, BertTokenizer
from tokenization import Tokenizer, load_chinese_base_vocab
import torch
import torch.nn as nn

word2idx, keep_token_ids = load_chinese_base_vocab('./chinese_wwm_ext_pytorch/vocab.txt', simplfied=True)
tokenizer = Tokenizer(word2idx)
print(tokenizer.encode('北京'))



config = BertConfig.from_pretrained('./chinese_wwm_ext_pytorch')
model = BertModel.from_pretrained(r'E:\Workspace\bert_crf\pretrained_model\chinese-bert-wwm')
embedding = nn.Embedding.from_pretrained(model.embeddings.word_embeddings.weight[keep_token_ids])
model.set_input_embeddings(embedding)
print(model.embeddings.word_embeddings.weight.shape)
print(embedding(torch.tensor([2, 1164, 674, 3])))
print(model(torch.tensor([[2, 1164, 674, 3]]))[0])


tokenizer = BertTokenizer.from_pretrained('./chinese_wwm_ext_pytorch')
IDS = tokenizer(['北京'], return_tensors='pt')
bert_model = BertModel.from_pretrained(r'E:\Workspace\bert_crf\pretrained_model\chinese-bert-wwm')
print(bert_model(**IDS)[0])