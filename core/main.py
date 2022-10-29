import os 
import time
import json
import openai
import spacy
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from datasets import load_dataset, load_from_disk
from transformers import DPRQuestionEncoder,DPRQuestionEncoderTokenizer
from utils import (
    CrossEncoder,
    read_json, 
    write_json, 
    get_ques_emb_batch,
    construct_prompt,
    chunks,
    chunk_size_helper,
    complete,
    make_test_instance,
    make_test_instance_only_edit
)

def get_gpt3_response(params, train_sentences, train_labels, test_sentences, return_all_prompts=False,
                       num_tokens_to_predict_override=None, override_prompt=None, temp = 0.7, max_tokens= 64):
    """
    Obtain model's responses on test sentences, given the training examples
    :param params: parameters for the experiment
    :param train_sentences: few-shot training sentences
    :param train_labels: few-shot training labels
    :param test_sentences: few-shot test sentences
    :param return_all_prompts: whether to return all the prompts
    :param num_tokens_to_predict_override: whether to override num token to predict
    :param override_prompt: whether to override prompt
    :return: a list of dictionaries
    """
    all_raw_answers = []

    # can optionally ignore the normal prompt and feed in a custom prompt (used for contextual calibration)
    if override_prompt is None:
        prompts = []
        for test_sentence in test_sentences:
            prompts.append(construct_prompt(params, train_sentences, train_labels, test_sentence))
    else:
        prompts = override_prompt

    num_tokens_to_predict = max_tokens

    chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
    for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
        resp = complete(test_chunk_prompts, num_tokens_to_predict, params['model'], temp = temp)
        for answer_id, answer in enumerate(resp['choices']):
            all_raw_answers.append(answer)
    if return_all_prompts:
        return all_raw_answers, prompts
    else:
        return all_raw_answers


def editor(args, retrieved_outputs=None):

    # ADD API KEY
    
    # with open(os.path.join('openai_key.txt'), 'r') as f:
    #     key = f.readline().strip()
    #     openai.api_key = key
    
    if not retrieved_outputs:
        with open(f'../{args.task}/prompt_only_edits.json', 'r') as f:
            params = json.load(f)
    else:
        with open(f'../{args.task}/prompt.json', 'r') as f:
            params = json.load(f)

    params['bs'] = args.bz
    params['model'] = 'text-davinci-002'
    params['task_format'] = 'nlg' 

    train_sentences = params['training_instances']
    train_labels = params['training_labels']

    edited_outputs = []

    nlp = spacy.load("en_core_web_sm")
    test_sentences =[]
    if retrieved_outputs:
        for row in tqdm(retrieved_outputs):
            test_sentence = make_test_instance(row, args.task, nlp)
            test_sentences.append(test_sentence)
    else:
        retrieved_outputs = read_json(args.input_file)
        for row in tqdm(retrieved_outputs):
            test_sentence = make_test_instance_only_edit(row, args.task)
            test_sentences.append(test_sentence)

    raw_test_answers = get_gpt3_response(params, 
                    train_sentences = train_sentences,
                    train_labels= train_labels,
                    test_sentences=test_sentences,
                    max_tokens = params['max_tokens']
                    )
    assert len(raw_test_answers) == len(retrieved_outputs)

    for row, ans in zip(retrieved_outputs, raw_test_answers):
        raw_test_answer = ans['text'].strip()
        row['edited'] = raw_test_answer
        edited_outputs.append(row)
    
    return edited_outputs




def reranker(args, retrieved_outputs):
    
    device = f'cuda:{args.gpuid}' if args.cuda else 'cpu' 
    cross_encoder_model = CrossEncoder(model_name = args.cross_checkpoint, 
                            max_length = 64,
                            device = device)
    querys = []
    ids= []
    labels =[]

    for i in retrieved_outputs:
        querys.append(i['premise'][0] + ' [SEP] ' + i['old_hypothesis'][0])
        labels.append(i['old_label'][0])

    BATCH_SIZE = args.bz
    top_k = 2
    output =[]


    for idk in tqdm(range(0,len(querys),BATCH_SIZE)):
        query = querys[idk: idk + BATCH_SIZE]
        label = labels[idk: idk + BATCH_SIZE]
        retrieved_examples = [i['new_hypothesis'][0] for i in retrieved_outputs[idk: idk+ BATCH_SIZE]]
        sentence_pairs = [[query[j], retrieved_examples[j][i]] for j in range(len(query)) for i in range(len(retrieved_examples[j]))]
        # SCORE
        ce_scores = cross_encoder_model.predict(sentence_pairs, batch_size = 512)
        prev = 0
        for i in range(len(query)):
            data = {'premise': [], 'old_hypothesis': [], 'new_hypothesis': [], 'old_label': []}
            length = len(retrieved_examples[i])
            hits = []
            for score, text in zip(ce_scores[prev: prev + length], retrieved_examples[i]):
                hits.append((text, score))
            prev += length
            hits = sorted(hits, key=lambda x: x[1], reverse=True)
            p, old_h = query[i].split(' [SEP] ')
            data['premise'].append(p)
            data['old_hypothesis'].append(old_h)
            data['new_hypothesis'].append(([h[0] for h in hits[:top_k]]))
            data['old_label'].append(label[i])
            output.append(data)

    return output
     



def retriever(args):
    data_embed = load_from_disk(args.corpus)
    data_embed.load_faiss_index('demo', args.index)
    device = f'cuda:{args.gpuid}' if args.cuda else 'cpu' 
    qmodel = DPRQuestionEncoder.from_pretrained(f'{args.cfdpr_checkpoint}/quencoder')
    qmodel.to(device)
    qtokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    data = read_json(args.input_file)
    querys = []
    labels =[]
    if args.task == 'nli':
        for i in data:
            querys.append(i['premise'] + ' [SEP] ' + i['hypothesis'])
            labels.append(i['label'])

    elif args.task == 'senti':
        for i in data:
            querys.append(i['text'])
            labels.append(i['label'])

    batch_size = args.bz

    output =[]
    _key = 'text' if args.task == 'nli' else 'sentence'
    
    for idk in tqdm(range(0,len(querys),batch_size)):
        query = querys[idk: idk + batch_size]
        label = labels[idk: idk + batch_size]
        question_embedding = get_ques_emb_batch(query, device, qmodel, qtokenizer)
        scores, retrieved_examples = data_embed.get_nearest_examples_batch('demo', question_embedding, k=20) # Top 50
        new_samples = []
        for i in retrieved_examples:
            new_samples.append(list(OrderedDict.fromkeys(i[_key])))
        for i in range(len(query)):
            if args.task == 'nli':
                data = {
                'premise': [],'old_hypothesis': [],'new_hypothesis': [],'old_label': []}
                p, h = query[i].split(' [SEP] ')
                data['premise'].append(p)
                data['old_hypothesis'].append(h)
                data['new_hypothesis'].append(new_samples[i])
                data['old_label'].append(label[i])

            elif args.task == 'senti':
                data = {
                    'text': [], 'old_label': [], 'retrieved': [],
                }
                data['text'].append(query[i])
                data['old_label'].append(label[i])
                data['retrieved'].append(new_samples[i])

            output.append(data)
    return output

def generate_counterfactuals(args):

    if args.strategy == 'retrieve':
        final_cf = retriever(args)
    elif args.strategy == 'edit':
        final_cf = editor(args)
    else:
        print('Retrieving Counterfactuals')
        retrieved_outputs = retriever(args)
        print('Retrieving Done')
        if args.do_reranking and args.task == 'nli':
            print('Re-ranking retrieved outpus')
            retrieved_outputs = reranker(args, retrieved_outputs)
        print('Editing')
        final_cf = editor(args, retrieved_outputs)
    
    write_json(final_cf, args.output_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--gpuid", type=int, default=0, help="gpu ID")
    parser.add_argument("--task", type=str, choices=['senti', 'nli'], help="task setting")
    parser.add_argument("--strategy", choices=['edit', 'retrieve', 'retrieve-edit'], type = str, help="augmentation strategy")
    parser.add_argument("--do_reranking", action="store_true", help="do reranking of retrieved outputs")
    parser.add_argument("--cross_checkpoint", type = str, help="trained cross encoder directory")    
    parser.add_argument("--cfdpr_checkpoint", type = str, help="trained retriever directory")
    parser.add_argument("--corpus", type = str, help="corpus directory")
    parser.add_argument("--index", type = str, help="index path")    
    parser.add_argument("--bz", default=20, type = int, help="batch size")    
    parser.add_argument("--input_file", type = str, help="input file list of jsons")
    parser.add_argument("--output_file", type = str, help="ouput file list of jsons")    
    args = parser.parse_args()

    generate_counterfactuals(args)