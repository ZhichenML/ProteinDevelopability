
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Develop
from sklearn.metrics import mean_squared_error
from tdc.utils import load as data_load
from sklearn.model_selection import train_test_split
from datasets import Dataset
from evaluate import load
import numpy as np


# metric = load("mse")

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     return metric.compute(predictions=predictions, references=labels)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}

def load_and_split_data(path='./data', name = 'tap'):
    label_list = retrieve_label_name_list('TAP')
    print('labels for multi-target: ', label_list)

    df = data_load.pd_load(name, path)
    df = df.dropna() 
    df2 = df.loc[:, ~df.columns.duplicated()]  ### remove the duplicate columns
    df = df2
    
    for ind in range(len(df)):
        v = df.loc[ind,'X']
        v = v.strip('[\']').split('\\n')
        v[0] = v[0].strip("'")
        v[1] = v[1].strip(" '")
        v = v[0] + v[1]
        df.loc[ind,'X'] = v
    
    sequences = df['X'].tolist()
    labels = df[label_list[-1]]
    assert len(sequences) == len(labels)

    
    return sequences, labels

sequences, labels = load_and_split_data()
train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.25, shuffle=True)

model_checkpoint = '/public/home/gongzhichen/hf_models/esm150m'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
train_tokenized = tokenizer(train_sequences)
test_tokenized = tokenizer(test_sequences)


train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(test_tokenized)
train_dataset = train_dataset.add_column("labels", train_labels)
test_dataset = test_dataset.add_column("labels", test_labels)


num_labels = max(train_labels + test_labels) + 1  # Add 1 since 0 can be a label
# model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, 
                                                           num_labels=1,
                                                           ignore_mismatched_sizes=True)

# set trainer arguments
model_name = model_checkpoint.split("/")[-1]
batch_size = len(train_dataset)

args = TrainingArguments(
    f"{model_name}-finetuned-localization",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="rmse",
    push_to_hub=False,
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
    
train, valid, test = load_and_split_data()


# input = 'QVKLQESGAELARPGASVKLSCKASGYTFTNYWMQWVKQRPGQGLDWIGAIYPGDGNTRYTHKFKGKATLTADKSSSTAYMQLSSLASEDSGVYYCARGEGNYAWFAYWGQGTTVTVSS'

# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# # model = AutoModel.from_pretrained(model_checkpoint)
# outputs = model(**tokenizer(input, return_tensors='pt'))

# import pdb; pdb.set_trace()
# outputs.last_hidden_state[:,0,:]


# train_tokenized = tokenizer(train_sequences)
# test_tokenized = tokenizer(test_sequences)




# batch_size = 32

# args = TrainingArguments(
#     evaluation_strategy = "epoch",
#     save_strategy = "epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=3,
#     report_to="none",
#     weight_decay=0.01,
#      output_dir='/content/drive/MyDrive/kaggle/',
#     metric_for_best_model='accuracy')



# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     rmse = mean_squared_error(labels, predictions, squared=False)
#     return {"rmse": rmse}


# trainer = Trainer(
#     model,
#     args,
#     train_dataset=tokenized_datasets['train'],
#     eval_dataset=tokenized_datasets['test'],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )