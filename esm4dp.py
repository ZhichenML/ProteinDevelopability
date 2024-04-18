
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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer, BertModel, BertTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments,
                          set_seed, AutoConfig, PreTrainedModel)
from transformers.integrations import TensorBoardCallback
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import logging
logging.set_verbosity_debug()
logger = logging.get_logger(__name__)

import torch
from torch import nn
from transformers import Trainer

from data import load_and_process_data

set_seed(42)

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        
        outputs=model(inputs.get('input_ids'),inputs.get('attention_mask'),inputs.get('labels'))
        
        outputs = model(**inputs)
        
        logits = outputs.get('logits')
        loss_fct = nn.MSELoss()
        
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
        




def compute_metrics_mse(eval_pred):
    metric = load("mse")
    r2 = load("r2")
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions, squared=True)
    r2 = r2_score(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    evs = explained_variance_score(labels, predictions)
    return {"mse": mse, 'r2': r2,'mae': mae, 'evs': evs}


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


class develop_esm(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained_path=None, dropout_rate=0.2, fc_hidden_size=640, use_keras_init=True, use_pooling=False, freeze_pretrained=True):
        super(develop_esm,self).__init__()
        # self.cfg=cfg
        # if config_path is None:
        #     self.config = AutoConfig.from_pretrained(cfg.model,output_hidden_states=True)
        # else:
        #     self.config = torch.load(config_path)
        # if pretrained_path is None:
        #     self.pretrained_model = AutoModel.from_pretrained(cfg.model,config=self.config)
        # else:
        self.pretrained_model = AutoModel.from_pretrained(pretrained_path, 
                                                          config=AutoConfig.from_pretrained(pretrained_path),
                                                          )
        if freeze_pretrained:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        self.dropout_rate = dropout_rate
        self.fc_dropout = nn.Dropout(dropout_rate)
        hidden_size = int(fc_hidden_size/10)
        self.fc = nn.Linear(fc_hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
        if use_keras_init:
          self._init_weights(self.fc)
        self.use_pooling=use_pooling
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            #module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data=torch.nn.init.xavier_uniform(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self,input_ids, attention_mask, labels=None): ##直接传入字典不需要其它的
        # inputs的输入为x，不要包含标签
        
        outputs=self.pretrained_model(input_ids, attention_mask,return_dict=True)
        # import pdb; pdb.set_trace() 
        # outputs=self.pretrained_model(**inputs, return_dict=True)
        if self.use_pooling:
          # use only cls hidden states
        #   preds=self.fc(self.fc_dropout(outputs.get('last_hidden_state')[:,0,:]))
          preds=self.fc_out(self.activation(self.fc(self.fc_dropout(torch.mean(outputs.get('last_hidden_state'), 1))))) 
        else:
          x = outputs.get('pooler_output')
          x = self.activation(self.fc(self.fc_dropout(x)))
          preds=self.fc_out(x)
        loss = None
        if labels is not None:
            
            loss_fct = nn.MSELoss()
            loss = loss_fct(preds.squeeze(), labels.squeeze())
        return TokenClassifierOutput(loss=loss,logits=preds, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
        
class get_esm_output():
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint
        self.pretrained_model = AutoModel.from_pretrained(model_checkpoint, 
                                                          config=AutoConfig.from_pretrained(model_checkpoint),
                                                          )
        self.tokenizer_ = AutoTokenizer.from_pretrained(model_checkpoint)
    
    def transform_all(self, sequences, labels, name='tap'):
        embedding = self.get_cls(sequences)
        tap = {'X': embedding, 'Y': labels}
        np.savez('/public/home/gongzhichen/code/data/'+ name +'_nosplit.npz', **tap)
        
    def transform(self, train_sequences, valid_sequences, test_sequences, train_labels, valid_labels, test_labels):
        # will be deprecated
        train_embedding = self.get_cls(train_sequences)
        valid_embedding = self.get_cls(valid_sequences)
        test_embedding = self.get_cls(test_sequences)
        tap = {'train_X': train_embedding, 'train_Y': train_labels, 'valid_X': valid_embedding, 'valid_Y': valid_labels, 'test_X': test_embedding, 'test_Y': test_labels}
        np.savez('/public/home/gongzhichen/code/data/tap.npz', **tap)
    
    def get_cls(self, seq_tuple):
        sequences, h_chain, l_chain = seq_tuple
        ret = []        
        for seq_h, seq_l in zip(h_chain, l_chain):
            token_seq_h = self.tokenizer_(seq_h, return_tensors='pt')
            token_seq_l = self.tokenizer_(seq_l, return_tensors='pt')
            outputs_h = self.pretrained_model(**token_seq_h)
            outputs_l = self.pretrained_model(**token_seq_l)
            outputs_h = outputs_h.last_hidden_state[:,0,:].detach().numpy().squeeze()
            outputs_l = outputs_l.last_hidden_state[:,0,:].detach().numpy().squeeze()
            ret.append(np.concatenate((outputs_h, outputs_l)))  ##这是获取得到的 cls 蛋白整体代表的向量
        return np.array(ret)
    
    def get_seq(self, seq_tuple):
        sequences, h_chain, l_chain = seq_tuple
        ret = []        
        for seq_h, seq_l in zip(h_chain, l_chain):
            token_seq_h = self.tokenizer_(seq_h, return_tensors='pt')
            token_seq_l = self.tokenizer_(seq_l, return_tensors='pt')
            outputs_h = self.pretrained_model(**token_seq_h)
            outputs_l = self.pretrained_model(**token_seq_l)
            outputs_h = outputs_h.last_hidden_state.detach().numpy().squeeze().T
            outputs_l = outputs_l.last_hidden_state.detach().numpy().squeeze().T
            ret.append(np.concatenate((outputs_h, outputs_l), 1))  ##这是获取得到的 cls 蛋白整体代表的向量
        
        return np.array(ret)

class funcs():
    def __init__(self):
        pass

    def get_embedding_all(self, name='tap'):
        print(f'start get_embedding_all, data_name: {name}')
        generator = get_esm_output(model_checkpoint)
        generator.transform_all(sequences, labels, name=name)
        print(f'end get_embedding_all {name}')

    def get_embedding(self):
        
  
        
        train_val_sequences, test_sequences, train_val_labels, test_labels = train_test_split(sequences, labels, test_size=0.2, shuffle=True)
        train_sequences, valid_sequences, train_labels, valid_labels = train_test_split(train_val_sequences, train_val_labels, test_size=0.1, shuffle=True)

        generator = get_esm_output(model_checkpoint)
        generator.transform(train_sequences, valid_sequences, test_sequences, train_labels, valid_labels, test_labels)
        
    def train_pipeline(self):
        
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

  
        
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

        
        do_train = True
        do_eval = True
        
        
        

        if do_train:
            # model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, 
            #                                                     num_labels=1,
            #                                                     ignore_mismatched_sizes=True)
            
            model = develop_esm(cfg=None, config_path=f'{model_checkpoint}/config.json', pretrained_path=model_checkpoint)
            args = TrainingArguments(
                f"{model_name}-finetuned-localization",
                evaluation_strategy = "epoch",
                save_strategy = "epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=500,
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model="mae",
                push_to_hub=False,
                auto_find_batch_size= False,
                logging_dir = "./log_files/", 
                logging_strategy='epoch',
                group_by_length = True,
                save_total_limit=10,
                seed=42,
                greater_is_better=False,
                remove_unused_columns=False,
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
            # model.save_pretrained('./output_model')
            torch.save(model.state_dict(), f'{model_name}-finetuned-localization/pytorch_model.pt')

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        
        
        
        
        if do_eval:
            latest_checkpoint = '/public/home/gongzhichen/code/ProteinDevelopability/esm150m-finetuned-localization/checkpoint-89'
            # model = AutoModelForSequenceClassification.from_pretrained(latest_checkpoint, num_labels=1, ignore_mismatched_sizes=True)
            
            model = develop_esm(cfg=None, config_path=f'{model_checkpoint}/config.json', pretrained_path=model_checkpoint)
            model.load_state_dict(torch.load(f'{model_name}-finetuned-localization/pytorch_model.pt'))
            model.eval()

            
            args = TrainingArguments(
                f"{model_name}-finetuned-localization",
                evaluation_strategy = "epoch",
                save_strategy = "epoch",
                learning_rate=2e-5, 
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=500,
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
                remove_unused_columns=True,
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
            
            # metrics = trainer.evaluate(eval_dataset=test_dataset)
            # # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            # metrics["eval_samples"] = len(test_dataset)
            # trainer.log_metrics("eval", metrics)
            # trainer.save_metrics("eval", metrics)
            
            predictions = trainer.predict(test_dataset=test_dataset)
            print(predictions)
            print('All done!')

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


if __name__ == '__main__':
    model_checkpoint = '/public/home/gongzhichen/hf_models/esm150m'
    data_path='/public/home/gongzhichen/code/data'
    data_name = 'tap' #'sabdab_chen'
    sequences, labels = load_and_process_data(path=data_path, name = data_name, label_ind=0)
    
    can_do = funcs()
    
    can_do.get_embedding_all(name=data_name)
    
    # can_do.train_pipeline()
