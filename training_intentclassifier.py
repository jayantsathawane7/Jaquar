# -*- coding: utf-8 -*-


import json
import codecs
intents = json.load(codecs.open('jaquar_intentsss.json', 'r', 'utf-8-sig'))

x,y = [],[]
for intent in intents["intents"]:
    patterns = set(intent["pattern"])
    for pattern in patterns:
        x.append(pattern.lower().strip())
        y.append(intent["tag"].lower().strip())

import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df = pd.DataFrame({"text":x,"labels":y})
df['labels'] = le.fit_transform(df['labels'])
df = df.sample(frac=1).reset_index(drop=True)
df.head()

# df.to_csv('jaquar_intents.csv', index=False)

from datasets import Dataset
dataset = Dataset.from_pandas(df)

number_of_labels = len(le.classes_)
number_of_labels

id2label = {i:label for i,label in enumerate(le.classes_)}
label2id = {label:i for i,label in id2label.items()}

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_classifier = dataset.map(preprocess_function, batched=True)

tokenized_classifier.set_format("torch",columns=["input_ids", "attention_mask", "labels"])

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_classifier, shuffle=True, batch_size=2, collate_fn=data_collator
)

for x in train_dataloader:
    print(x)
    break

from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
import torch.utils.checkpoint
import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoConfig, RobertaModel, RobertaPreTrainedModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IntentClassifier(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel.from_pretrained(
            'sentence-transformers/all-distilroberta-v1') #, add_pooling_layer=False
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)


    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self,input_ids=None,attention_mask=None,labels=None):
        output = self.roberta(input_ids=input_ids,attention_mask=attention_mask)
        pooler = self.mean_pooling(output, attention_mask)
        # pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        logits = self.classifier(pooler)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

config = AutoConfig.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
del config._name_or_path
config.id2label= id2label
config.label2id= label2id
config.num_classes = number_of_labels
config

model = IntentClassifier(config).to('cuda')

from torch.optim import AdamW

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct


optimizer = AdamW(model.parameters(), lr=5e-5)
EPOCHS = 50

def train(epoch, model, dataloader):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    
    for _,data in enumerate(dataloader, 0):

        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']

        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.logits, dim=1)
        n_correct += calcuate_accu(big_idx, labels)

        nb_tr_steps += 1
        nb_tr_examples+=labels.size(0)
        
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

model.train()
for epoch in range(EPOCHS):
    train(epoch,model,dataloader=train_dataloader)
model.eval()

from torch.nn.functional import softmax

texts = df['text'].to_list()
labels = df['labels'].to_list()

for text,label in zip(texts,labels):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt', return_attention_mask=True)
    with torch.no_grad():
        output = model(**tokens.to(device))
        scores = softmax(output[0], dim=1)[0].detach().cpu().numpy()
        # if scores.argmax()!=label:
        #     print(text, scores.max()*100)
        print(text.center(50), id2label[scores.argmax()].center(30),"-",id2label[label].center(30), scores.max()*100)

from torch.nn.functional import softmax
text = "prime nu"

encoded_input = tokenizer(text, return_tensors='pt')
encoded_input = encoded_input.to('cuda')
output = model(**encoded_input.to('cuda'))
scores = softmax(output[0], dim=1)[0].detach().cpu().numpy()
labels[scores.argmax()], scores.max()
print(text.center(30), id2label[scores.argmax()].center(30), scores.max()*100)

model.save_pretrained('./JaquarIntentClassification')
tokenizer.save_pretrained('./JaquarIntentClassification')

from transformers import AutoTokenizer
intent_tokenizer = AutoTokenizer.from_pretrained('JaquarIntentClassification')
intent_model = IntentClassifier.from_pretrained('JaquarIntentClassification').to('cuda')

from torch.nn.functional import softmax

texts = df['text'].to_list()
labels = df['labels'].to_list()

for text,label in zip(texts,labels):
    tokens = intent_tokenizer(text, padding=True, truncation=True, return_tensors='pt', return_attention_mask=True)
    with torch.no_grad():
        output = intent_model(**tokens.to(device))
        scores = softmax(output[0], dim=1)[0].detach().cpu().numpy()
        print(text.center(50), id2label[scores.argmax()].center(30),"-",id2label[label].center(30), scores.max()*100)

