from transformers import BertTokenizerFast, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification


def get_model(name):
    if name=='bert-base-uncased':
        model = BertForSequenceClassification.from_pretrained(name)
        return model
