import os 
import sys
import openai
import numpy as np
import time
import json
import torch
import random
import logging
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from tqdm import tqdm
from datasets import load_dataset
from collections import OrderedDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from typing import Dict, Type, Callable, List, Tuple
from torch.optim import Optimizer
from tqdm.autonotebook import tqdm, trange
from nltk.corpus import stopwords
global stop_words 
stop_words = set(stopwords.words('english'))
global punct
punct = ['.', '!', "?", '<', '>', '/', ',']



def read_json(path):
    data =[]
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_json(data, path):
    with open(path, 'w') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')

def get_ques_emb_batch(input_data, device,  model, tokenizer):
    data = tokenizer(input_data, truncation = True, padding ='max_length', max_length = 64,return_tensors="pt")
    data.to(device)
    return model(**data).pooler_output.detach().cpu().numpy()

def get_words_nli(list_of_outputs, nlp):
    words =[]
    for text in list_of_outputs:
        doc = nlp(text)
        for token in doc:
            #print(token.pos_, token.text)
            if token.pos_ in ['ADJ','ADP','ADV','NOUN','VERB'] and token.text.lower() not in words:
                words.append(token.text.lower()) 
    return words[:5]

def get_words_senti(list_of_outputs, nlp):
    words =[]
    for text in list_of_outputs:
        doc = nlp(text)
        for token in doc:
            x = token.text.lower()
            if x not in stop_words and x not in words and x not in punct:
                words.append(x) 
    return words


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chunk_size_helper(params):
    # Set the batch size (the size of the chunks determines the batch size). Default to 4 for GPT-2 and 20 for OpenAI if
    # no batch size is specified.
    bs = params['bs']
    if bs is None:
        assert params['model'] in ['ada', 'babbage', 'curie', 'text-davinci-002', 'ada-beta', 'babbage-beta']
        return 20
    else:
        return bs


def complete(prompt, l, model, temp=0, num_log_probs=None, echo=False, n=None):
    """complete the prompt using a language model"""
    assert l >= 0
    assert temp >= 0
    return complete_gpt3(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo, n=n)

def construct_prompt(params, train_sentences, train_labels, test_sentence):
    """construct a single prompt to be fed into the model"""
    # special case when the user defines a custom prompt function. 
    if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
        return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

    # take the prompt template and fill in the training and test example
    prompt = params["prompt_prefix"]
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]
    for s, l in zip(train_sentences, train_labels):
        prompt += q_prefix
        prompt += s + "\n"
        if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
            assert params['task_format'] == 'classification'
            l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
        else:
            assert isinstance(l, str) # string labels
            assert params['task_format'] == 'nlg'
            l_str = l

        prompt += a_prefix
        prompt += l_str + "\n\n"

    prompt += q_prefix
    prompt += test_sentence + "\n"
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
    return prompt

def complete_gpt3(prompt, l, model_name, temp=0, num_log_probs=None, echo=False, n=None):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.Completion.create(engine=model_name, prompt=prompt, max_tokens=l, temperature=temp,
                                                logprobs=num_log_probs, echo=echo, stop='\n', n=n)
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False

            print("API error:", error)
            time.sleep(1)
    return response


def make_test_instance_only_edit(test_sentence, task):
    s = ''
    if task == 'nli':
        old_label = "definitely False" if test_sentence['label'] == 2 else 'definitely True'
        new_label = "definitely True" if test_sentence['label'] == 2 else 'definitely False'
        s = s + 'original sentence 1: ' + test_sentence['premise'] + '\t'
        s = s + 'original sentence 2: ' + test_sentence['hypothesis'] + '\t'
        s = s + 'initial relationship label: ' + old_label + '\t'
        s = s + 'target label: ' + new_label + '\t'

    elif task == 'senti':
        label = "Positive" if test_sentence['label'] == 0 else "Negative"
        new_label = "Positive" if test_sentence['label'] == 1 else "Negative"
        s = s + 'Review: '+ test_sentence['text'] + '\n'
        s = s + 'Label: ' + label + '\n'
        s = s + 'Target Label: ' +  new_label + '\n'
    else:
        raise NotImplementedError
    
    return s


def make_test_instance(test_sentence, task, nlp):
    s = ''
    if task == 'nli':
        list_of_wds = get_words_nli(test_sentence['new_hypothesis'][0], nlp)
        old_label = "definitely False" if test_sentence['old_label'][0] == 2 else 'definitely True'
        new_label = "definitely True" if test_sentence['old_label'][0] == 2 else 'definitely False'
        s = s + 'original sentence 1: ' + test_sentence['premise'][0] + '\t'
        s = s + 'original sentence 2: ' + test_sentence['old_hypothesis'][0] + '\t'
        s = s + 'initial relationship label: ' + old_label + '\t'
        s = s + 'target label: ' + new_label + '\t'
        s = s + f'List of words: {list_of_wds}'

    elif task == 'senti':
        list_of_wds = get_words_senti(test_sentence['retrieved'][0], nlp)
        label = "Positive" if test_sentence['old_label'][0] == 0 else "Negative"
        new_label = "Positive" if test_sentence['old_label'][0] == 1 else "Negative"
        s = s + 'Review: '+ test_sentence['text'][0] + '\n'
        s = s + 'Label: ' + label + '\n'
        s = s + f'List of words: {list_of_wds}' + '\n'
        s = s + 'Target Label: ' +  new_label + '\n'
    else:
        raise NotImplementedError
    
    return s


def setup_for_distributed_mode(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: object,
    n_gpu: int = 1,
    local_rank: int = -1,
    fp16: bool = False,
    fp16_opt_level: str = "O1",
) -> (nn.Module, torch.optim.Optimizer):
    model.to(device)
    if fp16:
        try:
            import apex
            from apex import amp

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device if device else local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    return model, optimizer
def _infer_slurm_init(cfg) -> Tuple[str, int, int, int]:

    node_list = os.environ.get("SLURM_STEP_NODELIST")
    if node_list is None:
        node_list = os.environ.get("SLURM_JOB_NODELIST")

    if node_list is None:
        raise RuntimeError("Can't find SLURM node_list from env parameters")

    local_rank = None
    world_size = None
    distributed_init_method = None
    device_id = None
    try:
        hostnames = subprocess.check_output(["scontrol", "show", "hostnames", node_list])
        distributed_init_method = "tcp://{host}:{port}".format(
            host=hostnames.split()[0].decode("utf-8"),
            port=cfg['distributed_port'],
        )
        nnodes = int(os.environ.get("SLURM_NNODES"))
        ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
        if ntasks_per_node is not None:
            ntasks_per_node = int(ntasks_per_node)
        else:
            ntasks = int(os.environ.get("SLURM_NTASKS"))
            assert ntasks % nnodes == 0
            ntasks_per_node = int(ntasks / nnodes)

        if ntasks_per_node == 1:
            gpus_per_node = torch.cuda.device_count()
            node_id = int(os.environ.get("SLURM_NODEID"))
            local_rank = node_id * gpus_per_node
            world_size = nnodes * gpus_per_node
            # logger.info("node_id: %s", node_id)
        else:
            world_size = ntasks_per_node * nnodes
            proc_id = os.environ.get("SLURM_PROCID")
            local_id = os.environ.get("SLURM_LOCALID")
            local_rank = int(proc_id)
            device_id = int(local_id)

    except subprocess.CalledProcessError as e:  # scontrol failed
        raise e
    except FileNotFoundError:  # Slurm is not installed
        pass
    return distributed_init_method, local_rank, world_size, device_id

def setup_cfg_gpu(cfg):
    """
    Setup params for CUDA, GPU & distributed training
    """
    ws = os.environ.get("WORLD_SIZE")
    cfg['distributed_world_size'] = int(ws) if ws else 1

    if cfg['distributed_port'] and cfg['distributed_port'] > 0:
        init_method, local_rank, world_size, device = _infer_slurm_init(cfg)

        # logger.info(
        #     "Inferred params from SLURM: init_method=%s | local_rank=%s | world_size=%s",
        #     init_method,
        #     local_rank,
        #     world_size,
        # )

        cfg['local_rank'] = local_rank
        cfg['distributed_world_size'] = world_size
        cfg['n_gpu'] = 1

        torch.cuda.set_device(device)
        device = str(torch.device("cuda", device))

        torch.distributed.init_process_group(
            backend="nccl", init_method=init_method, world_size=world_size, rank=local_rank
        )

    elif cfg['local_rank'] == -1:  # single-node multi-gpu (or cpu) mode
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg['no_cuda'] else "cpu"))
        cfg['n_gpu'] = torch.cuda.device_count()
    else:  # distributed mode
        torch.cuda.set_device(cfg['local_rank'])
        device = str(torch.device("cuda", cfg['local_rank']))
        torch.distributed.init_process_group(backend="nccl")
        cfg['n_gpu'] = 1

    cfg['device'] = device
    return cfg



class CrossEncoder():
    def __init__(self, model_name:str, num_labels:int = None, max_length:int = None, device:str = None, tokenizer_args:Dict = {},
                  automodel_args:Dict = {}, default_activation_function = None):
        """
        A CrossEncoder takes exactly two sentences / texts as input and either predicts
        a score or label for this sentence pair. It can for example predict the similarity of the sentence pair
        on a scale of 0 ... 1.

        It does not yield a sentence embedding and does not work for individually sentences.

        :param model_name: Any model name from Huggingface Models Repository that can be loaded with AutoModel. We provide several pre-trained CrossEncoder models that can be used for common tasks
        :param num_labels: Number of labels of the classifier. If 1, the CrossEncoder is a regression model that outputs a continous score 0...1. If > 1, it output several scores that can be soft-maxed to get probability scores for the different classes.
        :param max_length: Max length for input sequences. Longer sequences will be truncated. If None, max length of the model will be used
        :param device: Device that should be used for the model. If None, it will use CUDA if available.
        :param tokenizer_args: Arguments passed to AutoTokenizer
        :param automodel_args: Arguments passed to AutoModelForSequenceClassification
        :param default_activation_function: Callable (like nn.Sigmoid) about the default activation function that should be used on-top of model.predict(). If None. nn.Sigmoid() will be used if num_labels=1, else nn.Identity()
        """
        # self.cfg = cfg
        # self.wandb = wandb_flag
        self.config = AutoConfig.from_pretrained(model_name)
        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config, **automodel_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        self.max_length = max_length
        # self.model, _ = setup_for_distributed_mode(
        #     self.model,
        #     None,
        #     cfg['device'],
        #     cfg['n_gpu'],
        #     cfg['local_rank'],
        # )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

        # if default_activation_function is not None:
        #     self.default_activation_function = default_activation_function
        #     try:
        #         self.config.sbert_ce_default_activation_function = util.fullname(self.default_activation_function)
        #     except Exception as e:
        #         logger.warning("Was not able to update config about the default_activation_function: {}".format(str(e)) )
        # # elif hasattr(self.config, 'sbert_ce_default_activation_function') and self.config.sbert_ce_default_activation_function is not None:
        # #     self.default_activation_function = util.import_from_string(self.config.sbert_ce_default_activation_function)()
        # else:

        self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()
            

    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                if isinstance(text, str):
                    texts[idx].append(text.strip())
                else:
                    texts[idx].append(str(text).strip())

            labels.append(example.label)

        tokenized = self.tokenizer(*texts, padding='max_length', truncation= True, return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels

    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(*texts, padding='max_length', truncation= True, return_tensors="pt", max_length=self.max_length)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized

    
    def predict(self, sentences: List[List[str]],
               batch_size: int = 128,
               show_progress_bar: bool = None,
               num_workers: int = 0,
               activation_fct = None,
               apply_softmax = False,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False
               ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.

        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor:  Conver the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, num_workers=num_workers, shuffle=False)

        # if show_progress_bar is None:
        #     show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)
        val_loss = 0
        with torch.no_grad():
            for features in iterator:
                model_predictions = self.model(**features, return_dict=True)
                #print(model_predictions.loss)
                #val_loss += model_predictions.loss.item()
                logits = activation_fct(model_predictions.logits)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores