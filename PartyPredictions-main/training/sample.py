import torch
from datasets import load_dataset, load_metric, Dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
import pandas as pd
import numpy as np

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
dropped_df.labels = dropped_df.labels//100 - 1 #what is this
#dropped_df['labels'].replace({100: 1, 200: 2}) # instead of prev line
#pd.to_numeric(dropped_df.labels)
dropped_df.replace([np.inf, -np.inf], np.nan, inplace=True)
dropped_df = dropped_df.dropna()
print(dropped_df.info())
# dropped_df.astype({'labels': int})  # figure it out then add it back

# log = open('log.txt', 'a')
# log.write('df read\n')

# sample df
dropped_df = dropped_df.sample(100)
# print("dropped_df.tail(): ", dropped_df.tail())

train_data = Dataset.from_pandas(dropped_df, preserve_index=False)
#print("train_data: ", train_data)

checkpoint = 'distilbert-base-uncased'  # used to be roberta-large
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(ex):
    return tokenizer(ex['text'], truncation=True, padding='max_length')


# DatasetDict
train_ds = train_data.train_test_split(test_size=0.5)  # 1/2 of set for test/val
test_valid = train_ds['test'].train_test_split(test_size=0.5)
'''
split_ds = DatasetDict({
    'train': train_ds['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']})
'''
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

'''
testset = split_ds['train'] # do it for the rest as necessary once i figure this out
print(testset.features)
new_features = testset.features.copy()
new_features["labels"] = ClassLabel(names=['democrat', 'republican'])
testset = testset.cast(new_features)
print(testset.features)
'''

# tokenize each set separately then put into dictionary?
tokenized_train = raw_train.map(tokenize_function, batched=True)
tokenized_test = raw_test.map(tokenize_function, batched=True)
tokenized_valid = raw_valid.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = DatasetDict({
    'train': tokenized_train.remove_columns('text'),
    'test': tokenized_test.remove_columns('text'),
    'validation': tokenized_valid.remove_columns('text')
})

tokenized_datasets.set_format('torch') #added
print("tokenized_datasets: ", tokenized_datasets)

print(tokenized_datasets['train']['labels'])

log.write('datasets tokenized\n')
# Training
training_args = TrainingArguments(
    "test-trainer",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
)  # evaluation strategy set

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
#model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)


def compute_metrics(eval_preds):
    metric = load_metric("accuracy")  # used to be glue, mnli
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# testing
print("tokenized_datasets['train']: ", tokenized_datasets['train'])

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
#log.write('up to training\n')
trainer.train()
#log.write('trained\n')
# saving
# save_dir = 'C:/Users/zz2953/Documents/Dorothy/models/'
save_dir = 'E:/zz2953/Dorothy/models/'
model.save_pretrained(save_dir + 'fine_tuned_distilbert_sample_10')
#log.write('saved\n')
#log.close()

# for 75-25 train-test/val split
# runtime for 40: 2:15
# runtime for 100: 3:59
# runtime for 200: 6:40
# runtime for 500: 14:55
# runtime for 1000: 28

