import torch
from tqdm.notebook import tqdm
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_prompts(model_name, model, tokenizer, tox_model, ppl, batch_size, 
                    max_continuation_length, max_prompt_length, output_dir, prompts_file, targeted_group=None, split=None):           
    """
    Collect the model continuations, toxicity, and sentiment for different groups of prompts.

    Args:
        model_name: str 
        model: transformers.PreTrainedModel
        tokenizer: transformers.PreTrainedTokenizer
        tox_model: transformers.PreTrainedModel
        ppl: float
        batch_size: int
        max_continuation_length: int
        max_prompt_length: int
        output_dir: str
        prompts_file: dict[str:dict[str:list[str]]]
        targeted_group: str
        split: str

    Returns:
        None
    """

    domain_results = []
    full_results = []
    for group, group_prompts in tqdm(prompts_file.items()):
        for title, prompts in group_prompts.items():
                
                if group != targeted_group:
                    continue      
                title = title.replace("_", " ").lower()
                generations, toxicity_scores = gen_prompt(
                    model, tokenizer, prompts, tox_model, batch_size, max_continuation_length, max_prompt_length
                )
                prompt_types=["original"]*len(prompts)
                                        
                domain_results.extend(
                    [
                        {
                            "group": group, 
                            "title": title,
                            "prompt": prompt_text,
                            "generation": gen,
                            "toxicity_score": tox_score,
                            "prompt_type": prompt_type,
                            "perplexity": ppl,
                            "split": split,
                        }
                        for gen, prompt_text, tox_score, prompt_type in zip(
                            generations, prompts, toxicity_scores, prompt_types
                        )
                    ]
                )
    
    full_results.extend(domain_results)
    full_results_pd = pd.DataFrame(full_results)
    full_results_pd.to_csv(output_dir + model_name + "_" + targeted_group + ".csv",index=False,)
    
def gen_prompt(
    model, tokenizer, data, tox_model, batch_size, max_continuation_length, max_prompt_length
):
    """
    Given some prompts, generate model continuation and measure both toxicity and sentiment scores.

    Args:
        model: transformers.PreTrainedModel
        tokenizer: transformers.PreTrainedTokenizer
        data: list[str]
        tox_model: transformers.PreTrainedModel
        batch_size: int
        max_continuation_length: int
        max_prompt_length: int

    Returns:
        outputs: list[str]
        toxicity_scores: list[float]
    """
    outputs, toxicity_scores = [], []

    for idx in tqdm(range(0, len(data), batch_size)):

        batch = data[idx : idx + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_length)
        print("idx is ", idx) 

        output_sequences = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            max_length=len(inputs["input_ids"][0]) + max_continuation_length,
            do_sample=True,
        )
        decoded_sequences = tokenizer.batch_decode(
            output_sequences, skip_special_tokens=True
        )
        print(decoded_sequences)
        toxicity_scores += tox_model.predict([x.replace("\n", " ").lower() for x in decoded_sequences])["toxicity"]
        outputs += decoded_sequences
        
    return outputs, toxicity_scores


def compute_ppl(model, tokenizer, stride, max_position_embeddings):
    """
    Compute perplexity of the model. Copied from https://huggingface.co/docs/transformers/perplexity

    Args:
        model: transformers.PreTrainedModel
        tokenizer: transformers.PreTrainedTokenizer
        stride: int
        max_position_embeddings: int
    
    Returns:
        ppl: float
    """
    ppl = {}
    for split in ["valid", "test"]:
        names = []    
        with open("./model/wikitext-2-raw-v1_" + split + ".txt", 'r') as fp:
            for line in fp:
                # remove linebreak from a current name
                # linebreak is the last character of each line
                x = line
                # add current item to the list
                names.append(x)

        encodings = tokenizer("".join(names) , return_tensors="pt")

        max_length = max_position_embeddings
        max_length=int(max_length/2)
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl[split] = round(float(torch.exp(torch.stack(nlls).mean())), 3)

    return ppl

