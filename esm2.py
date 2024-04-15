
# todo
# validation set in trainer
# tensorboard log

import os

os.chdir(os.path.dirname(__file__))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES']='6,7'

import numpy as np
from datasets import Dataset, load_dataset
from evaluate import load
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tdc.single_pred import Develop
from tdc.utils import load as data_load
from tdc.utils import retrieve_label_name_list
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer, BertModel, BertTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments,
                          set_seed)
from transformers.integrations import TensorBoardCallback
from transformers.utils import logging
logging.set_verbosity_info()

from torch import nn
from transformers import Trainer

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss


set_seed(42)

metric = load("mse")

def compute_metrics_mse(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions, squared=True)
    return {"mse": mse}

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

def Kfold_split(train_sequences, train_labels):
    # 存储每个 fold 的训练和验证集
    all_fold_trainX, all_fold_trainY, all_fold_validX, all_fold_validY = [], [], [], []
    # First make the kfold object
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    # import pdb; pdb.set_trace()
    # Now make our splits based off of the labels. 
    # We can use `np.zeros()` here since it only works off of indices, we really care about the labels
    splits = folds.split(train_sequences)
    # Finally, do what you want with it
    # In this case I'm overriding the train/val/test
    for train_idxs, val_idxs in splits:
        fold_trainX = np.array(train_sequences)[train_idxs].tolist()
        fold_trainY = np.array(train_labels)[train_idxs].tolist()
        fold_validX = np.array(train_sequences)[val_idxs].tolist()
        fold_validY = np.array(train_labels)[val_idxs].tolist()
        
        all_fold_trainX.append(fold_trainX)
        all_fold_trainY.append(fold_trainY)
        all_fold_validX.append(fold_validX)
        all_fold_validY.append(fold_validY)

    return all_fold_trainX, all_fold_trainY, all_fold_validX, all_fold_validY


if __name__ == '__main__':
    model_checkpoint = '/public/home/gongzhichen/hf_models/esm150m'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    sequences, labels = load_and_process_data()
    train_val_sequences, test_sequences, train_val_labels, test_labels = train_test_split(sequences, labels, test_size=0.2, shuffle=True)
    train_sequences, valid_sequences, train_labels, valid_labels = train_test_split(train_val_sequences, train_val_labels, test_size=0.1, shuffle=True)
    # all_fold_trainX, all_fold_trainY, all_fold_validX, all_fold_validY = Kfold_split(train_sequences, train_labels)
    
    train_tokenized = tokenizer(train_sequences)
    valid_tokenized = tokenizer(valid_sequences)
    test_tokenized = tokenizer(test_sequences)

    train_dataset = Dataset.from_dict(train_tokenized)
    valid_dataset = Dataset.from_dict(valid_tokenized)
    test_dataset = Dataset.from_dict(test_tokenized)
    train_dataset = train_dataset.add_column("labels", train_labels)
    valid_dataset = valid_dataset.add_column("labels", valid_labels)
    test_dataset = test_dataset.add_column("labels", test_labels)

    

    # set trainer arguments
    model_name = model_checkpoint.split("/")[-1]
    batch_size = len(train_dataset)

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, 
                                                            num_labels=1,
                                                            ignore_mismatched_sizes=True)
        
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
        metric_for_best_model="mse",
        push_to_hub=False,
        # do_train= True, 
        # do_eval= True, 
        # do_predict= True, 
        auto_find_batch_size= True,
        logging_dir = "./log_files/", 
        logging_strategy='epoch',
        group_by_length = True,
        save_total_limit=10,
        seed=42,
        greater_is_better=False,
        )



    early_stopping = EarlyStoppingCallback(early_stopping_patience= 10)

    trainer = RegressionTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping,]
    )
    

    do_train = False
    if do_train:
        
        
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
        model.save_pretrained('./output_model')

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    
    
    do_eval = True
    
    if do_eval:
        latest_checkpoint = '/public/home/gongzhichen/code/ProteinDevelopability/esm150m-finetuned-localization/checkpoint-89'
        model = AutoModelForSequenceClassification.from_pretrained(latest_checkpoint, 
                                                            num_labels=1,
                                                            ignore_mismatched_sizes=True)
        
        trainer = RegressionTrainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping,]
        )
        import pdb; pdb.set_trace()
        # metrics = trainer.evaluate(eval_dataset=test_dataset)
        metrics = trainer.predict(test_dataset=test_dataset)
        # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = len(test_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    # if training_args.do_predict:
    #     logger.info("*** Predict ***")
    #     predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

    #     max_predict_samples = (
    #         data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    #     )
    #     metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    #     trainer.log_metrics("predict", metrics)
    #     trainer.save_metrics("predict", metrics)

    #     predictions = np.argmax(predictions, axis=1)
    #     output_predict_file = os.path.join(training_args.output_dir, "predictions.txt")
    #     if trainer.is_world_process_zero():
    #         with open(output_predict_file, "w") as writer:
    #             writer.write("index\tprediction\n")
    #             for index, item in enumerate(predictions):
    #                 item = label_list[item]
    #                 writer.write(f"{index}\t{item}\n")


