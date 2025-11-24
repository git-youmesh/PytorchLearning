from torch import nn
from transformers import AutoConfig
from transformers import AutoTokenizer
import torch
from bertviz import head_view
from transformers import AutoModel
from math import sqrt
import torch.nn.functional as F
def scaled_dot_product_attention(query, key, value):
 dim_k = query.size(-1)
 scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
 weights = F.softmax(scores, dim=-1)
 return torch.bmm(weights, value)

text = "time flies like an arrow"
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)

inputs = tokenizer(text, return_tensors="pt",add_special_tokens=False)
inputs_embeds = token_emb(inputs.input_ids) # torch.Size([1, 5, 768])
query = key = value = inputs_embeds
scaled_dot_product_attention(query,key,value)
"""Our attention mechanism with equal query and key vectors will assign a
very large score to identical words in the context, and in particular to the
current word itself: the dot product of a query with itself is always 1. But in
practice, the meaning of a word will be better informed by complementary
words in the context than by identical words—for example, the meaning of
“flies” is better defined by incorporating information from “time” and
“arrow” than by another mention of “flies”. How can we promote this
behavior?
In practice,
the self-attention layer applies three independent linear transformations to
each embedding to generate the query, key, and value vectors. These
transformations project the embeddings and each projection(attention head) carries its own
set of learnable parameters, which allows the self-attention layer to focus on
different semantic aspects of the sequence

"""
class AttentionHead(nn.Module):
  def __init__(self, embed_dim, head_dim):
    super().__init__()
    """[batch_size, seq_len, head_dim]where head_dim is the
    number of dimensions we are projecting into
    """
    self.q = nn.Linear(embed_dim, head_dim)
    self.k = nn.Linear(embed_dim, head_dim)
    self.v = nn.Linear(embed_dim, head_dim)
  def forward(self, hidden_state):
      """ linear layers that apply matrix
multiplication to the embedding vectors"""

      attn_outputs = scaled_dot_product_attention(
      self.q(hidden_state), self.k(hidden_state),
      self.v(hidden_state))
      return attn_outputs
class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    embed_dim = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = embed_dim // num_heads
    self.heads = nn.ModuleList(
    [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
    )
    self.output_linear = nn.Linear(embed_dim, embed_dim)
  def forward(self, hidden_state):
    x = torch.cat([h(hidden_state) for h in self.heads],
    dim=-1)
    x = self.output_linear(x)
    return x

multihead_attn = MultiHeadAttention(config)
attn_output = multihead_attn(inputs_embeds) #torch.Size([1, 5, 768])



model = AutoModel.from_pretrained(model_ckpt,output_attentions=True)
sentence_a = "time flies like an arrow"
sentence_b = "fruit flies like a banana"
viz_inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt')
attention = model(**viz_inputs).attentions
sentence_b_start = (viz_inputs.token_type_ids == 0).sum(dim=1)
tokens = tokenizer.convert_ids_to_tokens(viz_inputs.input_ids[0])
head_view(attention, tokens, sentence_b_start, heads=[8])




