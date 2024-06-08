import tokenizers
import datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd

# checkpoint = 'E:/zz2953/Dorothy/models/trained_distilbert_pt'  # swapped for pytorch-trained version
checkpoint = 'E:/zz2953/Dorothy/models/fine_tuned_distilbert'
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')  # does this checkpoint need to match?

# encoding = tokenizer.encode_plus(text, add_special_tokens=True, truncation=True, padding="max_length", return_attention_mask=True, return_tensors="pt")

# evaluating accuracy
'''
def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


evaluate(model, df_test)
'''

pipe = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
# make df of questions
full_df = pd.read_csv('unique_questions.csv')
df = full_df.sample(20)
df = df['QuestionTxt'].to_frame()

df.info()
# create prediction column to add to df
predictions = []

# loop
# predict each question and add predicted label to df
for q in df['QuestionTxt']:
    pred_dict = pipe(q)[0]
    # predictions.append(pred_dict['label'])  # to just have highest label
    predictions.append(pred_dict)  # to show full amount for now

df['prediction'] = predictions

# df.to_csv('question_sample_predicted.csv')
print(df)
'''
pipe = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
print(pipe('Do you believe abortion is acceptable?'))
'''

# Time Trials:
# 25: 16.79 s
# 50: 19.42 s
# 100: 23.35 s
# 200: 38.64 s
# 400: 57.34 s
# 500: 1:09 m
# 1k: 2:19 m

