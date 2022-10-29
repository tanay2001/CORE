"""
This examples trains a CrossEncoder for the NLI task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it learns to predict the labels: "contradiction": 0, "entailment": 1, "neutral": 2.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_nli.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.utils import setup_cfg_gpu
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator, CEBinaryAccuracyEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import os
import gzip
import argparse
import pandas as pd
import csv
from tqdm import tqdm
import pickle
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument("--epochs", type = int, help="Num. of epochs")    
parser.add_argument("--bz", default=20, type = int, help="batch size")    
parser.add_argument("--train_file", type = str, help="input file list of jsons")
parser.add_argument("--val_file", type = str, help="ouput file list of jsons")    
args = parser.parse_args()


nli_dataset_path = args.train_file
nli_dataset_val = args.val_file


logger.info("Read NLI train dataset")

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
train_samples = []
dev_samples = []


df = pd.read_csv(nli_dataset_path)
for i in tqdm(range(df.shape[0])):
    row = df.loc[i]
    train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']))


df = pd.read_csv(nli_dataset_val)
for i in tqdm(range(df.shape[0])):
    row = df.loc[i]
    dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']))


train_batch_size = args.bz
num_epochs = args.epochs
model_save_path = 'output/training_nli-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

#Define our CrossEncoder model. We use distilroberta-base as basis and setup it up to predict 3 labels
model = CrossEncoder(model_name = 'bert-base-uncased',
                    device = 'cuda:0',
                    max_length =  128, 
                    num_labels=2)

#We wrap train_samples, which is a list ot InputExample, in a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

#During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev_samples, name='AllNLI-dev')


warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=2000,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          gradient_accumulation_steps = 1)


