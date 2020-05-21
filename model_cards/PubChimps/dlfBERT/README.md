# dlfBERT

A roBERTa-like classifier trained on sequence classification to differentiate between tensorflow and pytorch code. 

Model is finetuned from Hugging Face's [CodeBERTa-language-id](https://huggingface.co/huggingface/CodeBERTa-language-id) programming language classifier on 12,000+ examples of tensorflow and pytorch code, hosted on Box [here](https://ibm.box.com/shared/static/4x5bqo41yuwvvnce1tkx051khi726f11.npy
).

```python
from transformers import TextClassificationPipeline, RobertaTokenizer, RobertaForSequenceClassification

DLFBERT_LANGUAGE_ID = 'PubChimps/dlfBERT'
code = """
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
"""
pipeline = TextClassificationPipeline(
    model=RobertaForSequenceClassification.from_pretrained(DLFBERT_LANGUAGE_ID),
    tokenizer=RobertaTokenizer.from_pretrained(DLFBERT_LANGUAGE_ID)
)

pipeline(code)
