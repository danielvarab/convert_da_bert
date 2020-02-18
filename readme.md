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

# Below replace UNZIPPED_BERT_FOLDER with the unzipped folder downloaded from BotXO and PYTORCH_MODEL_FOLDER with the created output folder (e.g. `bert-base-danish-uncased-v1 or `bert-base-danish-uncased-v2)

# Note that there are multiple bert_model.ckpt files (don't mistakenly include the suffixes.)
transformers-cli convert \
    --model_type bert \
    --tf_checkpoint UNZIPPED_BERT_FOLDER/bert_model.ckpt \
    --config UNZIPPED_BERT_FOLDER/bert_config.json \
    --pytorch_dump_output PYTORCH_MODEL_FOLDER/pytorch_model.bin

# Copy the BERT configuration and vocabulary to the output folder
cp UNZIPPED_BERT_FOLDER/bert_config.json PYTORCH_MODEL_FOLDER/config.json
cp UNZIPPED_BERT_FOLDER/vocab.txt PYTORCH_MODEL_FOLDER/

```

## Using the model for MaskedLM
```python
import torch
import transformers

# Assuming the output folder above was "bert-base-danish-uncased-v2", and the folder is located in the same directory.
MODEL_FOLDER = "bert-base-danish-uncased-v2"

model = transformers.BertForMaskedLM.from_pretrained(MODEL_FOLDER)
tokenizer = transformers.BertTokenizer.from_pretrained(MODEL_FOLDER)
_ = model.eval()

txt = "[CLS] Jeg [MASK] dig . [SEP]"
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

