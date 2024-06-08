import torch
from torch.utils.data import DataLoader
from datasets import load_metric, Dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AdamW, get_scheduler
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

train_df = pd.read_csv('train.csv', encoding='latin1', low_memory=False)
dropped_df = train_df.drop(labels=['bonica.rid',
                                   'congno',
                                   'NID',
                                   'source',
                                   'date',
                                   'nominate_dim2',
                                   'recipient.cfscore',
                                   'bioguide_id',
                                   'nominate_dim1',
                                   'bioname',
                                   'state_abbrev',
                                   'district_code',
                                   'icpsr',
                                   'congress',
                                   'chamber',
                                   'Unnamed: 0'], axis=1)

dropped_df = dropped_df.rename(columns={"party_code": "labels"})
dropped_df = dropped_df[dropped_df.labels != 328]
dropped_df.labels = dropped_df.labels // 100 - 1  # Convert 100 and 200 to 0 and 1
dropped_df.replace([np.inf, -np.inf], np.nan, inplace=True)
dropped_df = dropped_df.dropna()

train_data = Dataset.from_pandas(dropped_df, preserve_index=False)

checkpoint = 'distilbert-base-uncased'  # used to be roberta-large
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(ex):
    return tokenizer(ex['text'], truncation=True, padding='max_length')


# DatasetDict
train_ds = train_data.train_test_split(test_size=0.5)  # 1/2 of set for test/val
test_valid = train_ds['test'].train_test_split(test_size=0.5)
split_ds = DatasetDict({
    'train': train_ds['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']})

# turn labels into ClassLabel
raw_train = train_ds['train']
new_features_train = raw_train.features.copy()
new_features_train["labels"] = ClassLabel(names=['democrat', 'republican'])
raw_train = raw_train.cast(new_features_train)

raw_test = test_valid['test']
new_features_test = raw_test.features.copy()
new_features_test["labels"] = ClassLabel(names=['democrat', 'republican'])
raw_test = raw_test.cast(new_features_test)

raw_valid = test_valid['train']
new_features_valid = raw_valid.features.copy()
new_features_valid["labels"] = ClassLabel(names=['democrat', 'republican'])
raw_valid = raw_valid.cast(new_features_valid)

for ds in [raw_train, raw_test, raw_valid]:
    print(ds.features)

# Tokenize each set separately then put into DatasetDict
tokenized_train = raw_train.map(tokenize_function, batched=True)
tokenized_test = raw_test.map(tokenize_function, batched=True)
tokenized_valid = raw_valid.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = DatasetDict({
    'train': tokenized_train.remove_columns('text'),
    'test': tokenized_test.remove_columns('text'),
    'validation': tokenized_valid.remove_columns('text')
})

tokenized_datasets.set_format('torch')  # Added
print("tokenized_datasets: ", tokenized_datasets)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# training using pure pytorch
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

optimizer = AdamW(model.parameters(), lr=3e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)  # added model assignment

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


metric = load_metric("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()

#save_dir = 'E:/zz2953/Dorothy/models/'
save_dir = '/scratch/zz2953/'  # for jubail
model.save_pretrained(save_dir + 'trained_distilbert_pt')
