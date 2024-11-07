import icl_dataset_loading
import selectionp
import torch

import copy
import random
import warnings
import itertools
import argparse
from tqdm import tqdm
from typing import List

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model_path", required=True)
    parser.add_argument("--compression_model_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max_token_len", help='max token length for ICL demos', default=750)
    parser.add_argument("--compression_rate", default=0.1)
    parser.add_argument("--device", default='cuda')
    args = parser.parse_args()
    return args

class PromptGenerator(torch.utils.data.Dataset):
    def __init__(self, dataset: dict,
        tokenizer, 
        num_plaintext_demonstrations: int, 
        num_softprompt_demonstrations: List[int], 
        seed: int,
        delimiter="\n\n", 
        content_free_string="N/A"
    ):
        """
        Initializes a PromptGenerator object.
        Properties:
        self.dataset: dict
        self.tokenizer: transformers.PreTrainedTokenizer
        self.num_plaintext_demonstrations: int
        self.num_softprompt_demonstrations: list[int]
        self.delimiter: str
        self.content_free_string: str
        self.all_softprompts_demonstrations_tokens: list[torch.Tensor]
        self.plaintext_demonstrations_tokens: torch.Tensor
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_plaintext_demonstrations = num_plaintext_demonstrations # int
        self.num_softprompt_demonstrations = num_softprompt_demonstrations # list[int] 
        self.delimiter = delimiter
        self.content_free_string = content_free_string
        # prevents collisions and avoids "incremental" sampling
        random.seed(10**6 * seed + 10**3 * sum(num_softprompt_demonstrations) + num_plaintext_demonstrations) 
        # sample indices for softprompt and plaintext demonstrations
        sample_idxs = random.sample(range(len(self.dataset["train"])), sum(num_softprompt_demonstrations) + num_plaintext_demonstrations)
        softprompt_idxs = sample_idxs[:sum(num_softprompt_demonstrations)]
        plaintext_idxs = sample_idxs[sum(num_softprompt_demonstrations):]
        if sum(self.num_softprompt_demonstrations) > 0: # if softprompt demonstrations are needed
            # splitting all softprompt demonstrations into chunks based on num_softprompt_demonstrations
            softprompt_examples = self.dataset["train"][softprompt_idxs]
            softprompt_examples = iter([dict(zip(softprompt_examples, i)) for i in zip(*softprompt_examples.values())]) # unzip dict
            chunked_softprompt_examples = [list(itertools.islice(softprompt_examples, 0, i)) for i in num_softprompt_demonstrations] 
            chunked_softprompt_demonstrations_tokens = []
            chunked_softprompt_demonstration_counts = []
            add_special_tokens = True   # adds start token only to the first chunk
            for chunk in chunked_softprompt_examples:
                softprompt_demonstrations_tokens, chunked_softprompt_demonstration_count = \
                    self.get_demonstrations_tokens(chunk, add_special_tokens=add_special_tokens)
                chunked_softprompt_demonstrations_tokens.append(softprompt_demonstrations_tokens)
                chunked_softprompt_demonstration_counts.append(chunked_softprompt_demonstration_count)
                add_special_tokens = False
            self.all_softprompts_demonstrations_tokens = chunked_softprompt_demonstrations_tokens # list of torch.Tensor
            self.num_softprompt_demonstrations = chunked_softprompt_demonstration_counts # revised list of int
        if self.num_plaintext_demonstrations > 0: # if plaintext demonstrations are needed
            plaintext_examples = self.dataset["train"][plaintext_idxs]
            plaintext_examples = [dict(zip(plaintext_examples, i)) for i in zip(*plaintext_examples.values())] # unzip dict
            self.plaintext_demonstrations_tokens, self.num_plaintext_demonstrations = \
                self.get_demonstrations_tokens(plaintext_examples, add_special_tokens=(sum(self.num_softprompt_demonstrations) == 0))
    def get_demonstration_string(self, example: dict, label=None, include_label=True, for_calibration=False) -> str:
        """
        Returns a demonstration string for a given example.

        example: dict
        label: int
        include_label: bool
        for_calibration: bool
        returns: str
        """
        example = copy.deepcopy(example)
        example["label"] = label if label is not None else example["label"]     # override label
        example["answer"] = example["options"][example["label"]] if include_label else ""
        if for_calibration:
            for input_key in self.dataset["input_keys"]:
                example[input_key] = self.content_free_string
        demonstration_string = self.dataset["template"].format(**example).rstrip()
        return demonstration_string
    def get_demonstrations_tokens(self, examples: list, add_special_tokens: bool, max_tokens=float('inf')):
        """
        Tokenizes demonstrations and returns the tokens and the number of examples that were used to create them (constrained by max_tokens).
        examples: list of dicts
        add_special_tokens: bool
        max_tokens: int
        returns: demonstrations_tokens: torch.Tensor, num_examples: int
        """
        demonstrations_string = ""
        num_examples = 0
        # keep adding examples until max_tokens is reached
        for example in examples:
            demonstration_string = self.get_demonstration_string(example) + self.delimiter
            extended_demonstrations_string = demonstrations_string + demonstration_string
            extended_demonstrations_tokens = self.tokenizer.encode(extended_demonstrations_string, add_special_tokens=add_special_tokens)
            if len(extended_demonstrations_tokens) <= max_tokens:
                demonstrations_string = extended_demonstrations_string
                num_examples += 1       
            else:
                break
        demonstrations_tokens = self.tokenizer.encode(demonstrations_string, add_special_tokens=add_special_tokens, return_tensors="pt")
        return demonstrations_tokens, num_examples
    def get_calibration_nlls(self, example: dict, model, device, is_ac: bool, softprompt=None, plaintext_demonstrations_tokens=None):
        """
        Computes the calibration NLLs for a given example.
        example: dict
        model: transformers.AutoModelForCausalLM | auto_compressor.AutoCompressorModel | auto_compressor.LlamaAutoCompressorModel
        device: torch.device
        is_ac: bool
        softprompt: torch.Tensor
        plaintext_demonstrations_tokens: torch.Tensor
        returns: calibration_nlls: torch.Tensor
        """
        assert (sum(self.num_softprompt_demonstrations) == 0) or (softprompt is not None)
        assert (self.num_plaintext_demonstrations == 0) or (plaintext_demonstrations_tokens is not None)
        add_special_tokens = ((self.num_plaintext_demonstrations + sum(self.num_softprompt_demonstrations)) == 0)
        unanswered_example_string = self.get_demonstration_string(example, include_label=False, for_calibration=True)
        unanswered_example_tokens = self.tokenizer.encode(unanswered_example_string, add_special_tokens=add_special_tokens, return_tensors="pt").to(device)
        calibration_nlls = []
        for label_idx in range(len(example["options"])):
            answered_example_string = self.get_demonstration_string(example, label=label_idx, for_calibration=True)
            answered_example_tokens = self.tokenizer.encode(answered_example_string, add_special_tokens=add_special_tokens, return_tensors="pt").to(device)
            option_tokens = answered_example_tokens[:,unanswered_example_tokens.shape[1]:]
            option_length = option_tokens.shape[1]
            plaintext_tokens = answered_example_tokens if plaintext_demonstrations_tokens is None else \
                torch.cat([plaintext_demonstrations_tokens, answered_example_tokens], dim=1)
            with torch.no_grad():
                calibration_option_logits = model.forward(plaintext_tokens, softprompt=softprompt, use_cache=False)["logits"][:,-option_length-1:-1,:] \
                    if is_ac else model.forward(plaintext_tokens, use_cache=False)["logits"][:,-option_length-1:-1,:]
                calibration_log_softmax = torch.log_softmax(calibration_option_logits, dim=-1)
                calibration_nll = -torch.mean(calibration_log_softmax.gather(dim=2, index=option_tokens.unsqueeze(-1)))
                calibration_nlls.append(calibration_nll)
        return torch.tensor(calibration_nlls)
    def __len__(self):
        return len(self.dataset["test"])
    def __getitem__(self, index: int) -> dict:
        """
        Returns a dictionary containing the following keys:
            answered_example_options: list of torch.Tensor 
            answer_options: list of torch.Tensor
            answer_idx: int
            test_example: dict
        index: int
        returns: dict
        """
        test_example = self.dataset["test"][index]
        add_special_tokens = ((self.num_plaintext_demonstrations + sum(self.num_softprompt_demonstrations)) == 0)
        unanswered_example_string = self.get_demonstration_string(test_example, include_label=False)
        unanswered_example_tokens = self.tokenizer.encode(unanswered_example_string, add_special_tokens=add_special_tokens, return_tensors="pt")
        answered_example_options_tokens = []
        options_tokens = []
        for label_idx in range(len(test_example["options"])):
            answered_example_string = self.get_demonstration_string(test_example, label=label_idx)
            answered_example_tokens = self.tokenizer.encode(answered_example_string, add_special_tokens=add_special_tokens, return_tensors="pt")
            option_tokens = answered_example_tokens[:,unanswered_example_tokens.shape[1]:]
            answered_example_options_tokens.append(answered_example_tokens)
            options_tokens.append(option_tokens)
        return_dict = {
            "answered_example_options": answered_example_options_tokens, # full answered demonstration alternatives
            "answer_options": options_tokens, # just the answers' alternatives
            "answer_idx": test_example["label"], # correct answer index
            "test_example": test_example # original test example
        }
        return return_dict

def getdemo(seed,num,ds,tokenizer):
    prompt_generator = PromptGenerator(
            dataset=ds, 
            tokenizer=tokenizer, 
            num_plaintext_demonstrations=num, 
            num_softprompt_demonstrations=[0], # list
            seed=seed
        )
    return prompt_generator

def getdemoNls(ds, tokenizer, max_len=750):
    demoNls = []
    # run for 4 trials with different seed and average the result
    for seed in tqdm(range(0,5)):
        if(seed==0): demoN = 1
        else: 
            if(demoN!=1): demoN = demoN-1
        prompt_generator = getdemo(seed, demoN, ds, tokenizer)
        while(prompt_generator.plaintext_demonstrations_tokens.shape[1]<max_len):
            demoN+=1
            prompt_generator = getdemo(seed, demoN, ds, tokenizer)
        if(demoN!=1 and abs(max_len-getdemo(seed, demoN-1, ds, tokenizer).plaintext_demonstrations_tokens.shape[1])<abs(max_len-prompt_generator.plaintext_demonstrations_tokens.shape[1])):
            demoN-=1
            prompt_generator = getdemo(seed, demoN, ds, tokenizer)
        demoNls.append(demoN)
    return demoNls

def compress_tk(model, tk, k):
    p = model(tk)['p']
    be1 = torch.topk(p,k)
    return (tk[0][sorted(be1.indices[0].to('cpu').tolist())].unsqueeze(0),torch.sort(be1.indices).values,p) 

def eval(prompt_generator, ds, model, plaintext_demonstrations_tokens, use_plaintext_demonstrations=True, device='cuda', use_calibration=True, position_ids=None):
    use_calibration = use_calibration
    device = device
    is_ac = False
    use_plaintext_demonstrations = use_plaintext_demonstrations
    softprompt = None
    plaintext_demonstrations_tokens = plaintext_demonstrations_tokens.to(device) if use_plaintext_demonstrations else None
    
    if use_calibration and not ds["recalibrate_every"]: 
        calibration_nlls = prompt_generator.get_calibration_nlls(
            ds["test"][0], 
            model, device, is_ac, 
            softprompt=softprompt, 
            plaintext_demonstrations_tokens=plaintext_demonstrations_tokens
        )
    
    num_correct = 0
    num_total = 0
    skip = False # flag for skipping examples that are too long
    
    progress_bar = tqdm(prompt_generator, mininterval=0)
    for example in progress_bar:
        if use_calibration and ds["recalibrate_every"]:
            calibration_nlls = prompt_generator.get_calibration_nlls(
                example["test_example"], 
                model, device, is_ac, 
                softprompt=softprompt, 
                plaintext_demonstrations_tokens=plaintext_demonstrations_tokens
            )
        
        conditioned_nlls = []
        # iterate over all candidate answer options
        batch_size = 8
        for option_idx in range(int(len(example["answer_options"])/batch_size)+1):
            tmp = example["answered_example_options"][option_idx*batch_size:(option_idx+1)*batch_size]
            max_length = max([l.shape[1] for l in tmp])
            new_tmp = [torch.nn.functional.pad(t, (0, max_length-t.shape[1])) for t in tmp]
            new_tmp = torch.cat(new_tmp, dim=0)
            answered_example_tokens = new_tmp.to(device)
            option_tokens = example["answer_options"][option_idx*batch_size:(option_idx+1)*batch_size]#.to(device)
            option_length = [o.shape[1] for o in option_tokens]
            plaintext_tokens = answered_example_tokens if plaintext_demonstrations_tokens is None else \
                torch.cat([plaintext_demonstrations_tokens.repeat((new_tmp.shape[0],1)), answered_example_tokens], dim=1)
            
            # if (not is_ac) and (plaintext_tokens.shape[-1] > 2048):
            #     warnings.warn("Input longer than 2048 tokens. Skipping example!")
            #     skip = True
            #     continue
        
            with torch.no_grad():
                if(position_ids!=None):
                    conditioned_answer_logits = model.forward(plaintext_tokens, position_ids=position_ids, softprompt=softprompt, use_cache=False)["logits"][:,-option_length-1:-1,:] \
                    if is_ac else model.forward(plaintext_tokens, position_ids=position_ids, use_cache=False)["logits"][:,-option_length-1:-1,:]
                else:
                    l = model.forward(plaintext_tokens, use_cache=False)["logits"] 
                    conditioned_answer_logits = [l[i,-v:,:]for i,v in enumerate(option_length)]
                    #conditioned_answer_logits = model.forward(plaintext_tokens, position_ids=position_ids, softprompt=softprompt, use_cache=False)["logits"][:,-option_length-1:-1,:] \
                    #if is_ac else model.forward(plaintext_tokens, use_cache=False)["logits"][:,-option_length-1:-1,:]
                for idx,c in enumerate(conditioned_answer_logits):
                    c = c.unsqueeze(0)
                    conditioned_log_softmax = torch.log_softmax(c, dim=-1)
                    conditioned_nll = -torch.mean(conditioned_log_softmax.gather(dim=2, index=option_tokens[idx].unsqueeze(-1).to(device)))
                    conditioned_nlls.append(conditioned_nll)
        
        if skip: 
            skip = False # reset flag
            continue
    
        conditioned_nlls = torch.tensor(conditioned_nlls) - calibration_nlls if use_calibration else torch.tensor(conditioned_nlls)
        nll_answer = torch.argmin(conditioned_nlls).item()
        num_correct += int(nll_answer == example["answer_idx"])
        num_total += 1
        progress_bar.set_postfix({"accuracy": num_correct / num_total}, refresh=False)
            
    print("Accuracy:", num_correct / num_total)
    if(plaintext_demonstrations_tokens!=None):
        print("#token:", plaintext_demonstrations_tokens.shape[1])
    return num_correct / num_total

def main(args):
    model, tokenizer = selectionp.load(args.compression_model_path)
    model = model.to(args.device)
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model_path).to(args.device)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    ds = icl_dataset_loading.get_ds(args.dataset)
    dls = getdemoNls(ds, tokenizer, args.max_token_len)
    
    result = []
    # run for 4 trials with different seed and average the result
    for i in range(1,5): 
        prompt_generator = getdemo(i,dls[i],ds, tokenizer)
        tmp = prompt_generator.plaintext_demonstrations_tokens.to('cpu')
        # compression
        compressed_plaintext_demonstrations_tokens, ids, p = compress_tk(model, tmp, int(tmp.shape[1]*args.compression_rate))
        txtc = tokenizer.decode(compressed_plaintext_demonstrations_tokens[0]).replace('<s>','')
        # pass to target model for inference
        compressed = target_tokenizer(txtc, return_tensors='pt').input_ids
        result.append(eval(prompt_generator, ds, target_model, compressed, device=args.device, use_calibration=True))
    print(f"Accuracy on {args.dataset}: {sum(result)/4}")

if __name__ == "__main__":
    main(read_args())