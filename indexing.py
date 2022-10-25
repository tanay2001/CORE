import argparse
from datasets import load_dataset, load_from_disk, Dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer,DPRQuestionEncoder
import torch
import torch.nn as nn
import os 
import faiss
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
import json
from nltk.tokenize import sent_tokenize

def _label(x):
    k = []
    for i in x['overall']:
        if i > 3:
            k.append(1)
        else:
            k.append(0)   
    x['overall'] = k
    return x

def encode(data, model, tokenizer):
    data = tokenizer(data, return_tensors='pt', truncation = True, padding ='max_length',max_length = 128) 
    data.to('cuda')
    with torch.no_grad():
        output = model(**data).pooler_output.cpu().numpy()
    return output

        
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
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("gloo", rank=1, world_size=3)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device if device else local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    return model, optimizer


def get_model_tokenizer(model_name):
    ctx_encoder = DPRContextEncoder.from_pretrained(model_name)
    tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    return ctx_encoder, tokenizer


def encode_corpus(data_embed, model, tokenizer, args):
    if args.do_encode:
        data_embed = data_embed.map(lambda x : {'embeddings': encode(x['text'], model = model, tokenizer=tokenizer)}, batched = True, batch_size = 1024)
        data_embed.save_to_disk(args.encode_path)
    if args.do_index:
        if not args.do_encode:
            data_embed = load_dataset(args.encode_path)
        d = args.index_dim 
        quantizer = faiss.IndexFlatIP(d)
        # Quantizing Parameters Used -> feel free to change them
        index = faiss.IndexIVFFlat(quantizer, d, args.n_clusters)
        index.nprobe = args.nprobe
        data_embed.add_faiss_index(column='embeddings',custom_index = index, train_size = args.train_size, index_name = 'demo') # 90000
        data_embed.save_faiss_index('demo', args.index_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--n_gpu", type=int, 
                    default=1,required = True, help="no. of gpus")

    # FAISS PARAMS
    parser.add_argument("--nprobe", type = int, 
                    default=30, help="faiss params")
    parser.add_argument("--d", type = int, 
                    default=768, help="faiss params")
    parser.add_argument("--n_clusters", type = int, 
                    default= 300,  help="faiss params")  
    parser.add_argument("--train_size", type = int, 
                    default= 90000,  help="faiss params")  

    parser.add_argument("--index_path", type = str, 
                        help="file path to save the index")
    parser.add_argument("--encode_path", type=str, 
                    help="file path to save the corpus encodings")
    parser.add_argument("--data_dir", type = str, 
                        help="data directory, ovewites the encode path")
    parser.add_argument("--model_dir", type = str,
                        required = True, help="CFDPR model checkpoint directory")
    parser.add_argument("--do_encode", required = True, action="store_true", help="")
    parser.add_argument("--do_index", required = True, action="store_true", help="")
    args = parser.parse_args()

    # sanity checks
    if args.do_index:
        assert args.index_path is not None and args.encode_path is not None
        if os.path.exists(args.index_path):
            print('Warning overwriting index')
    if args.do_encode:
        assert args.encode_path is not None
        if os.path.exists(args.encode_path):
            print('Warning overwriting encodings')
    


    # Setup models
    model, tokenizer = get_model_tokenizer(args.model_dir)
    if args.n_gpu >1:
        model, _ = setup_for_distributed_mode(model, None, 'cuda', args.n_gpu)
    
    dataset = load_from_disk(args.data_dir)

    if args.do_encode or args.do_index:
        encode_corpus(dataset, model, tokenizer, args)
    else:
        print('Nothing to do!')
        



