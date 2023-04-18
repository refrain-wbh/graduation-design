from accelerate import Accelerator

accelerator = Accelerator()
from transformers import (
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)
from tqdm import tqdm
import torch
import datasets

sst2_train = datasets.load_dataset("glue", "sst2", split="train")
sst2_valid = datasets.load_dataset("glue", "sst2", split="validation")
sst2_train_loader = torch.utils.data.DataLoader(sst2_train, batch_size=32, shuffle=True)
sst2_valid_loader = torch.utils.data.DataLoader(sst2_valid, batch_size=32)


config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=2)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", config=config
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

model, optimizer, sst2_train_loader, sst2_valid_loader = accelerator.prepare(
    model, optimizer, sst2_train_loader, sst2_valid_loader
)
model.train()
for epoch in range(4):
    for inputs in tqdm(sst2_train_loader):
        labels = inputs["label"]
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
