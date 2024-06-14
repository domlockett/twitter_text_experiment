from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from pickle import TRUE
import pandas as pd
from sqlalchemy import Column, Integer, MetaData, Table, create_engine, String, BigInteger,Text, update, and_, select, func, types,Float
import os

checkpoint = 'C:/Users/Path/to/Data/PartyPredictions-main/models/fine_tuned_distilbert'  # Path to model goes here.
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')  # Tokenizer can be adjusted if needed.
pipe = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, truncation=True)
conn = create_engine('sqlite:///C:/Users/Path/to/Data/data_22/processed/jan_clean.db').connect()
df = pd.read_sql(sql =" SELECT * FROM 'jan-o'", con = conn)
df['question'] =df.text_o.str.encode('latin1') 

predictions_o = []
scores_o = []



# Predict each question and add predictions to column lists.
for q in df['question']:  # Replace 'question' with the proper column name.
    q = str(q)  # As a precaution.
    pred_dict = pipe(q)[0]  # Gives dictionary of label and score.
    label = pred_dict['label']

    # This part is optional to make the labels more readable.
    # Replace names/shorthands as necessary.
    if label == 'LABEL_0':
        predictions_o.append('D')
    elif label == 'LABEL_1':
        predictions_o.append('R')
    else:
        predictions_o.append('N')  # In case of an invalid label.
    # Entire preceding part can be replaced by predictions.append(label)

    scores_o.append(pred_dict['score'])
#pd.DataFrame(predictions_o, columns=['bias_prediction']).to_csv('original_kauf_preds')

conn.close()
conn = create_engine('sqlite:///C:/Users/Path/to/Data/data_22/processed/jan_clean.db').connect()
df = pd.read_sql(sql =" SELECT * FROM 'jan'", con = conn)
df['question'] =df.text.str.encode('latin1', 'ignore') 
#df = pd.read_csv('questions.csv', encoding='latin1')  # Include file name and proper encoding here.
# Create lists to store information for new columns.
predictions = []
scores = []
for q in df['question']:  # Replace 'question' with the proper column name.
    q = str(q)  # As a precaution.
    pred_dict = pipe(q)[0]  # Gives dictionary of label and score.
    label = pred_dict['label']

    # This part is optional to make the labels more readable.
    # Replace names/shorthands as necessary.
    if label == 'LABEL_0':
        predictions.append('D')
    elif label == 'LABEL_1':
        predictions.append('R')
    else:
        predictions.append('N')  # In case of an invalid label.
    # Entire preceding part can be replaced by predictions.append(label)

    scores.append(pred_dict['score'])
