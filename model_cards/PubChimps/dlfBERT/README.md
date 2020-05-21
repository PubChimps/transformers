# dlfBERT

A roBERTa-like classifier trained on sequence classification to differentiate between tenosorflow and pytorch code.

Model is finetuned from Hugging Face's CodeBERTa-small-v1 programming language classifier on 12,000+ examples of tensorflow and pytorch code, hosted on Box [here](https://ibm.box.com/shared/static/4x5bqo41yuwvvnce1tkx051khi726f11.npy
).

```python
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained("PubChimps/dlfBERT")
model = TFRobertaForSequenceClassification.from_pretrained("PubChimps/dlfBERT",from_pt=True)

code = """
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
"""

data = tokenizer.encode(code, max_length=512, return_tensors='tf')
model.predict(data)
```
