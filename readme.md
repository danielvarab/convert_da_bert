# Convert Danish BERT from BotXO

Find the Danish BERT (recently renamed nordic BERT) model trained by BotXO [here](https://github.com/botxo/danish_bert) and follow the dropbox link.

## Prerequisites
1. Download the zipped BERT model.
2. Unzip the zip-folder.
3. Create a folder for the converted pytorch model (something like `bert-base-danish-uncased-v1` or `bert-base-danish-uncased-v2` depending on the version downloaded).

## Instructions to convert Tensorflow model to Pytorch (compatible with Huggingface)


```bash
# Install the huggingface library and tensorflow to convert
pip install transformers
pip install tensorflow
conda install pytorch torchvision cpuonly -c pytorch

# download botxo's danish model (v2).
# this was the dropbox link as of 11/6/2020
wget https://www.dropbox.com/s/19cjaoqvv2jicq9/danish_bert_uncased_v2.zip\?dl\=1

# unzip model. The zip includes a folder called 'danish_bert_uncased_v2'
unzip danish_bert_uncased_v2.zip

# output directory
mkdir bert-base-danish-uncased-v2

# Note that there are multiple bert_model.ckpt files (don't mistakenly include the suffixes.)
transformers-cli convert \
    --model_type bert \
    --tf_checkpoint danish_bert_uncased_v2/bert_model.ckpt \
    --config danish_bert_uncased_v2/bert_config.json \
    --pytorch_dump_output bert-base-danish-uncased-v2/pytorch_model.bin

# Copy the BERT configuration and vocabulary to the output folder
cp danish_bert_uncased_v2/bert_config.json bert-base-danish-uncased-v2/config.json
cp danish_bert_uncased_v2/vocab.txt bert-base-danish-uncased-v2/

```

## Using the model for MaskedLM
```python
import torch
from transformers import AutoModel, AutoTokenizer

# Assuming the output folder above was "bert-base-danish-uncased-v2", and the folder is located in the same directory.
MODEL_FOLDER = "bert-base-danish-uncased-v2"

model = AutoTokenizer.from_pretrained(MODEL_FOLDER)
tokenizer = AutoModel.from_pretrained(MODEL_FOLDER)
_ = model.eval()

txt = "[CLS] Jeg [MASK] dig . [SEP]"
tokens = tokenizer.tokenize(txt)

mask_index = tokens.index("[MASK]")

indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)

tokens_tensor = torch.tensor([indexed_tokens])
segments_ids = [0] * len(indexed_tokens)
segments_tensors = torch.tensor([segments_ids])

with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

# Top predicted token
predicted_index = torch.argmax(predictions[0, mask_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
print(predicted_token)

# Print the top k=10 predicted words
for i, idx in enumerate(torch.topk(predictions[0, mask_index], k=10)[1]):
    print("%d:" % (i+1), tokenizer.convert_ids_to_tokens(idx.item()))


```

