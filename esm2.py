
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Develop
from sklearn.metrics import mean_squared_error
from tdc.utils import load

def data_loader(path='./data', name = 'tap'):
    df = load.pd_load(name, path)
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
    
  
    # label_list = retrieve_label_name_list('TAP')
    # print('labels for multi-target: ', label_list)
    # data = Develop(name = 'TAP', label_name = label_list[3])

    # split = data.get_split()

    # train = split['train']
    # valid = split['valid']
    # test = split['test']
    

    # for v in test['Antibody'].tolist():
    #     v = v.strip('[\']').split('\\n')
    #     v[0] = v[0].strip("'")
    #     v[1] = v[1].strip(" '")
    #     v = v[0] + v[1]

    import pdb; pdb.set_trace()
    return train, valid, test


train, valid, test = data_loader()

model_checkpoint = '/public/home/gongzhichen/hf_models/esm150m'
input = 'QVKLQESGAELARPGASVKLSCKASGYTFTNYWMQWVKQRPGQGLDWIGAIYPGDGNTRYTHKFKGKATLTADKSSSTAYMQLSSLASEDSGVYYCARGEGNYAWFAYWGQGTTVTVSS'

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, 
                                                           num_labels=1,
                                                           ignore_mismatched_sizes=True)
# model = AutoModel.from_pretrained(model_checkpoint)
outputs = model(**tokenizer(input, return_tensors='pt'))

import pdb; pdb.set_trace()
outputs.last_hidden_state[:,0,:]


train_tokenized = tokenizer(train_sequences)
test_tokenized = tokenizer(test_sequences)




batch_size = 32

args = TrainingArguments(
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    report_to="none",
    weight_decay=0.01,
     output_dir='/content/drive/MyDrive/kaggle/',
    metric_for_best_model='accuracy')



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)