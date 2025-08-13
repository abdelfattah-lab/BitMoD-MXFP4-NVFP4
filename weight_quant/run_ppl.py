import torch
import torch.nn as nn

from datasets import load_dataset

import argparse
from tqdm import tqdm
from loguru import logger
import os
import random
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

from utils import (
    load_model_and_tokenizer, 
    add_common_args, 
    set_seed,
    model2path
)
from quantize import quant_model

    
@torch.no_grad()
def eval_ppl(model, tokenizer, args):
    results = {}
    for task_eval in args.datasets:
        if task_eval == "wikitext":
            # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
            testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
            model.seq_len = args.seq_len
            testenc = testenc.input_ids.to(model.device)
            nsamples = testenc.numel() // model.seq_len
            nlls = []
            loss_fct = nn.CrossEntropyLoss()
            for i in tqdm(range(nsamples), desc="evaluating..."):
                batch = testenc[:, (i * model.seq_len) : ((i + 1) * model.seq_len)].to(
                    model.device
                )
                with torch.no_grad():
                    lm_logits = model(batch).logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = testenc[
                    :, (i * model.seq_len) : ((i + 1) * model.seq_len)
                ][:, 1:]
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * model.seq_len
                nlls.append(neg_log_likelihood.item())

            ppl = torch.exp(torch.tensor(nlls).sum() / (nsamples * model.seq_len))
            results["wikitext"] = ppl.item()
            print(f'Wikitext-2 perplexity: {ppl.item()}')
            print('\n')

        elif task_eval == "c4":
            model_net = model_name_or_path.split('/')[-1]
            model_family = '_'.join(model_net.lower().split('-')[:-1])
            model.seq_len = args.seq_len

            cache_testloader = f'./data_cache/testloader_{model_family}_c4_{args.seq_len}.cache'
            os.makedirs(os.path.dirname(cache_testloader), exist_ok=True)
            if os.path.exists(cache_testloader):
                testenc = torch.load(cache_testloader)
                print(f"load calibration from {cache_testloader}")
            else:
                valenc = []
                testenc = load_dataset("allenai/c4", data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split="validation")
                for _ in range(256): # run 256 samples
                    while True:
                        i = random.randint(0, len(testenc) - 1)
                        tmp = tokenizer(testenc[i]['text'], return_tensors='pt')
                        if tmp.input_ids.shape[1] > (model.seq_len+1):
                            break
                    i = random.randint(0, tmp.input_ids.shape[1] - model.seq_len - 1)
                    j = i + model.seq_len
                    valenc.append(tmp.input_ids[:, i:j])
                testenc = torch.hstack(valenc)
                torch.save(testenc, cache_testloader)
            
            nsamples = testenc.numel() // model.seq_len
            loss_fct = nn.CrossEntropyLoss()
            nlls = []
            with tqdm(range(nsamples)) as progress:
                for i in progress:
                    batch = testenc[:, (i * model.seq_len) : ((i + 1) * model.seq_len)].to(model.device)
                    with torch.no_grad():
                        lm_logits = model(batch, use_cache=False, output_hidden_states=False, output_attentions=False)[0]
                    shift_logits = lm_logits[:, :-1, :].contiguous().float()
                    shift_labels = testenc[:, (i * model.seq_len) : ((i + 1) * model.seq_len)][:, 1:].to(model.device)
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                    )
                    neg_log_likelihood = loss.float() * model.seq_len
                    nlls.append(neg_log_likelihood.item())
                    progress.set_description(f"Evaluating")

            ppl = torch.exp(torch.tensor(nlls).sum() / (nsamples * model.seq_len))
            results['c4'] = ppl.item()
            print(f'C4 perplexity: {ppl.item()}')
            print('\n')

    return results
    

if __name__ == '__main__':
    # Ignore all warnings
    warnings.filterwarnings("ignore")
    # Set random seed
    set_seed(0)

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument('--datasets', type=lambda s: [item for item in s.split(',')], default=['wikitext'], help="Task to be evaled")
    parser.add_argument('--seq_len', type=int, help='sequence length for ppl evaluation', default=2048)
    parser.add_argument("--output_dir", type=str, default="results/ppl", help="output directory")
    args = parser.parse_args()  
    
    model_name = args.model_name
    model_name_or_path = model2path[model_name]
    w_bits      = args.w_bits
    w_dtype     = args.w_dtype
    w_groupsize = args.w_groupsize

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")
    logger.info(f"#################### Model Info ####################")
    logger.info(f"* Model: {model_name_or_path}")
    logger.info(f"* Datasets: {args.datasets}")
    logger.info(f"* Sequence length {args.seq_len}")
    logger.info(f"#################### Start evaluating ppl with the following configurations: ####################")
    logger.info(f"* Bench compression!!!")
    logger.info(f"* Weights Bits:       {w_bits}")
    logger.info(f"* Weights Data Type:  {w_dtype}")
    logger.info(f"* Weights Group Size: {w_groupsize}")
    logger.info("")

    logger.info("#################### Creating output directory ... ####################")
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    if w_bits == 16:
        output_file_name = "Baseline_FP16"
    elif ("mxfp4" in w_dtype) or ("nvfp4" in w_dtype):
        output_file_name = f"w{w_bits}-{w_dtype}"
    else:
        output_file_name = f"w{w_bits}-{w_dtype}-g{w_groupsize}"

    output_file_path = os.path.join(output_dir, f"{output_file_name}.txt")
    # check if result file exists
    if os.path.isfile(output_file_path):
        print(f'Found existing output file {output_file_name} for this experiment. Exit!\n\n')
        exit()
    print(f'Results will be saved to the output file: {output_file_name}\n')
    
    logger.info("#################### Loading model and tokenizer ... ####################")
    model, tokenizer = load_model_and_tokenizer(model_name)
    quant_model(model, wq_bits=w_bits, wq_datatype=w_dtype, wq_groupsize=w_groupsize)
    logger.info("")

    logger.info("#################### Start running perplexity evaluation ... ####################")
    res = eval_ppl(model, tokenizer, args)

    # Save results to JSON file
    with open(output_file_path, "w") as f:
        for dataset, ppl in res.items():
            logger.info(f"{dataset} PPL: {ppl}")
            f.write(f"{dataset.ljust(10)} PPL: {ppl}\n")
    
    print(f"Results saved to {output_file_path} \n\n")
