from transformers import pipeline

classifier = pipeline("sentiment-analysis")

print(classifier("I am Very Happy to Share this Tutorial of this HuggingFace Library 🤗 to You."))

results = classifier(["I am Very Happy to Share this Tutorial of this HuggingFace Library 🤗 to You.", "I Hope You Dont Hate it."])

for result in results:
    print(f"label: {result["label"]}, with score: {result["score"]}")

'''You can see the second sentence has been classified as negative (it needs to be positive or negative)
but its score is fairly neutral.By default, the model downloaded for this pipeline
is called "distilbert-base-uncased-finetuned-sst-2-english".'''

# dataset to classify mutlilanguages english, french, german, dutch, italian and spanish
classifier = pipeline("sentiment-analysis", model = "nlptown/bert-base-multilingual-uncased-sentiment")

# i translated "food is bad but you are kind" into french with google translate
# https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment to understand about library, scores and star system
print(classifier("la nourriture est mauvaise mais tu es gentil"))

# whats happening in backend process is tokenizing and model working on text without stopwords
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
# This model only exists in PyTorch, so we use the `from_pt` flag to import that model in TensorFlow.
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(model_name) # tokenizer according to that model only
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
print(classifier("I Am a Good Boy."))



model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
inputs = tokenizer("I am Very Happy to Share this Tutorial of this HuggingFace Library 🤗 to You.")
print(inputs)

tf_batch = tokenizer(["I am Very Happy to Share this Tutorial of this HuggingFace Library 🤗 to You.", "I Hope You Dont Hate it."], padding=True, truncation=True, max_length=512, return_tensors="tf")

for key, value in tf_batch.items():
    print(f"{key}: {value.numpy().tolist()}")