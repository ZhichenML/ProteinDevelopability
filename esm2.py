
# todo
# validation set in trainer
# tensorboard log

import os
os.chdir(os.path.dirname(__file__))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES']='4,6,7'

from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback
from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Develop
from sklearn.metrics import mean_squared_error
from tdc.utils import load as data_load
from sklearn.model_selection import train_test_split
from datasets import Dataset
from evaluate import load
import numpy as np


metric = load("mse")

def compute_metrics_mse(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}

def load_and_process_data(path='/public/home/gongzhichen/data', name = 'tap'):
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
    labels = df[label_list[-1]].tolist()
    assert len(sequences) == len(labels)

    
    return sequences, labels

sequences, labels = load_and_process_data()
import pdb; pdb.set_trace()
train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.2, shuffle=True)

model_checkpoint = '/public/home/gongzhichen/hf_models/esm150m'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
train_tokenized = tokenizer(train_sequences)
test_tokenized = tokenizer(test_sequences)


train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(test_tokenized)
train_dataset = train_dataset.add_column("labels", train_labels)
test_dataset = test_dataset.add_column("labels", test_labels)


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
    num_train_epochs=400,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="rmse",
    push_to_hub=False,
    do_train= True, 
    do_eval= True, 
    do_predict= True, 
    auto_find_batch_size= True,
    logging_dir = "./log_files/", 
    logging_strategy='epoch',
    group_by_length = True,
    save_total_limit=10,
    seed=42,
    greater_is_better=False,
)



early_stopping = EarlyStoppingCallback(early_stopping_patience= 10, 
                                    early_stopping_threshold= 0.001,
                                    )

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping,]
)

resume_from_checkpoint = None
last_checkpoint = None
checkpoint = None
if resume_from_checkpoint is not None:
    checkpoint = resume_from_checkpoint
elif last_checkpoint is not None:
    checkpoint = last_checkpoint
train_result = trainer.train(resume_from_checkpoint=checkpoint)
metrics = train_result.metrics

trainer.save_model()  # Saves the tokenizer too for easy upload

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
# trainer.evaluate(eval_dataset=)



