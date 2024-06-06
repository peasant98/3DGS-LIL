from transformers import DistilBertTokenizer, DistilBertModel
import torch
import torch.nn as nn


if __name__ == '__main__':
    action = "move to"
    
    
    # Load pre-trained model tokenizer and model
    # Step 1: Obtain the Phrase Embedding using DistilBERT
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    inputs = tokenizer(action, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state

    phrase_embedding = torch.mean(embeddings, dim=1)

    print(phrase_embedding.shape)