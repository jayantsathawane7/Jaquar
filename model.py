from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch
from transformers import RobertaModel, RobertaPreTrainedModel

class IntentClassifier(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config=config)

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
