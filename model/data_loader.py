import pandas as pd
from transformers import BertTokenizer, RobertaTokenizer
import torch
import numpy as np


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        encodings,
        encodings_gender_swap=None,
        encodings_gender_blind=None,
        gender_swap=None,
        weights=None,
        labels=None,
    ):
        self.encodings = encodings
        self.encodings_gender_swap = encodings_gender_swap
        self.encodings_gender_blind = encodings_gender_blind
        self.gender_swap = gender_swap
        self.labels = labels
        self.weights = weights

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def data_loader(
    dataset,
    max_length,
    classifier_model,
    IPTTS=None,
    apply_data_augmentation=None,
    apply_data_substitution=None,
    data_augmentation_ratio=1,
    data_substitution_ratio=0.5,
):
    """
    Load the data  from the CSV files and an object for each split in the dataset.
    args:
        dataset: the dataset used
        max_length: The maximum length of the sentences that we classify (in terms of the number of tokens)
        classifier_model: the model name
        IPTTS: a flag that means that we want to also return the IPTTS dataset,
            which is the dataset on which we compute the bias metrics. We can also
            compute the AUC score on it.
    returns:
        the function returns objects, for the training, validation and test
        datasets. Each object contains the tokenized data and the corresponding
        labels.
    """
    data_train = pd.read_csv("./data/" + dataset + "_train_original_gender.csv", engine="python",
        error_bad_lines=False,)
    # The gender swap means that we flip the gender in each example in out dataset.
    # For example, the sentence "he is a doctor" becomes "she is a doctor".
    data_train_gender_swap = pd.read_csv("./data/" + dataset + "_train_gender_swap.csv", engine="python",
        error_bad_lines=False,) 
    data_train_gender_blind = pd.read_csv("./data/" + dataset + "_train_gender_blind.csv", engine="python",
        error_bad_lines=False,)          

    data_valid = pd.read_csv("./data/" + dataset + "_valid_original_gender.csv", engine="python",
        error_bad_lines=False,)
    data_valid_gender_swap = pd.read_csv("./data/" + dataset + "_valid_gender_swap.csv", engine="python",
        error_bad_lines=False,)
    data_valid_gender_blind = pd.read_csv("./data/" + dataset + "_valid_gender_blind.csv", engine="python",
        error_bad_lines=False,)        

    data_test = pd.read_csv("./data/" + dataset + "_test_original_gender.csv", engine="python",
        error_bad_lines=False,)      
    # These weights are some scores that are given to the examples to reflect their imporance, based on https://arxiv.org/pdf/2004.14088.pdf
    train_weights = torch.from_numpy(np.load("./data/weights_" + dataset + "_all_seeds_train.npy"))
    valid_weights = torch.from_numpy(np.load("./data/weights_" + dataset + "_all_seeds_valid.npy"))
        
    if apply_data_augmentation:
        # In data augmentation, we double the size of the training data by adding
        # the gender-fliped example of every training example.
        number_of_original_examples = len(data_train_gender_swap)
        data_train = pd.concat(
            [
                data_train,
                data_train_gender_swap.iloc[
                    0 : int(data_augmentation_ratio * number_of_original_examples)
                ],
            ],
            axis=0,
            ignore_index=True,
        )        

        data_train_gender_swap = pd.concat(
            [
                data_train_gender_swap,
                data_train.iloc[
                    0 : int(data_augmentation_ratio * number_of_original_examples)
                ],
            ],
            axis=0,
            ignore_index=True,
        )

    if apply_data_substitution:
        # We substitute each example with the gender-flipped one, with a probability of 0.5,
        # as described in https://arxiv.org/abs/1909.00871
        for i in range(len(data_train)):
            if np.random.uniform(0, 1, 1)[0] < data_substitution_ratio:
                temp = data_train_gender_swap[data_train_gender_swap.columns[0]].iloc[i]
                data_train_gender_swap[data_train_gender_swap.columns[0]].iloc[
                    i
                ] = data_train[data_train.columns[0]].iloc[i]
                data_train[data_train.columns[0]].iloc[i] = temp

    model_name = classifier_model
    if model_name in [
        "bert-base-cased",
        "bert-large-cased",
        "distilbert-base-cased",
        "bert-base-uncased",
        "bert-large-uncased",
        "distilbert-base-uncased",
    ]:
        tokenizer = BertTokenizer.from_pretrained(
            "./saved_models/cached_tokenizers/" + model_name
        )
    elif model_name in ["roberta-base", "distilroberta-base"]:
        tokenizer = RobertaTokenizer.from_pretrained(
            "./saved_models/cached_tokenizers/" + model_name
        )      

    # ----- 1. Preprocess data -----#
    # Preprocess data
    X_train = list(data_train[data_train.columns[0]])
    y_train = list(data_train[data_train.columns[1]])

    X_val = list(data_valid[data_valid.columns[0]])
    y_val = list(data_valid[data_valid.columns[1]])

    X_test = list(data_test[data_test.columns[0]])
    y_test = list(data_test[data_test.columns[1]])    

    X_train_gender_swap = list(data_train_gender_swap[data_train_gender_swap.columns[0]])
    X_val_gender_swap = list(data_valid_gender_swap[data_valid_gender_swap.columns[0]])

    X_train_gender_blind = list(data_train_gender_blind[data_train_gender_blind.columns[0]])
    X_val_gender_blind = list(data_valid_gender_blind[data_valid_gender_blind.columns[0]])

    # This is a boolean tensor that indentifies the examples that have undergone gender swapping
    train_gender_swap = torch.tensor(
        data_train[data_train.columns[0]]
        != data_train_gender_swap[data_train_gender_swap.columns[0]]
    )
    valid_gender_swap = torch.tensor(
        data_valid[data_valid.columns[0]]
        != data_valid_gender_swap[data_valid_gender_swap.columns[0]]
    )  


    X_train_tokenized = tokenizer(
        X_train, padding=True, truncation=True, max_length=max_length
    )
    X_train_tokenized_gender_swap = tokenizer(
        X_train_gender_swap, padding=True, truncation=True, max_length=max_length
    ) 
    X_train_tokenized_gender_blind = tokenizer(
        X_train_gender_blind, padding=True, truncation=True, max_length=max_length
    )      

    X_val_tokenized = tokenizer(
        X_val, padding=True, truncation=True, max_length=max_length
    )
    X_val_tokenized_gender_swap = tokenizer(
        X_val_gender_swap, padding=True, truncation=True, max_length=max_length
    ) 
    X_val_tokenized_gender_blind = tokenizer(
        X_val_gender_blind, padding=True, truncation=True, max_length=max_length
    )     

    X_test_tokenized = tokenizer(
        X_test, padding=True, truncation=True, max_length=max_length
    )    

    train_dataset = Dataset(
        encodings=X_train_tokenized,
        encodings_gender_swap=X_train_tokenized_gender_swap,
        encodings_gender_blind=X_train_tokenized_gender_blind,
        labels=y_train,
        gender_swap = train_gender_swap,
        weights = train_weights,
    )
    val_dataset = Dataset(
        encodings=X_val_tokenized,
        encodings_gender_swap=X_val_tokenized_gender_swap,
        encodings_gender_blind=X_val_tokenized_gender_blind,
        labels=y_val,
        gender_swap = valid_gender_swap,
        weights = valid_weights,
    )    
    test_dataset = Dataset(
        encodings=X_test_tokenized,
        labels=y_test,
    )

    # IPTTS is a synthetic dataset that is used to compute the fairness metrics
    if IPTTS:
        IPTTS_all = pd.read_csv("./data/" + "IPTTS_split.csv",engine="python",
        error_bad_lines=False,)
        IPTTS_valid = IPTTS_all[IPTTS_all["split"] == "valid"].reset_index(drop=True)

        X_IPTTS_all = list(IPTTS_all[IPTTS_all.columns[0]])
        X_IPTTS_valid = list(IPTTS_valid[IPTTS_valid.columns[0]])

        y_IPTTS_all = list(IPTTS_all["Class"])
        y_IPTTS_valid = list(IPTTS_valid["Class"])

        X_IPTTS_all_tokenized = tokenizer(
            X_IPTTS_all, padding=True, truncation=True, max_length=max_length
        )
        X_IPTTS_valid_tokenized = tokenizer(
            X_IPTTS_valid, padding=True, truncation=True, max_length=max_length
        )          

        IPTTS_all_dataset = Dataset(
            encodings=X_IPTTS_all_tokenized,
            labels=y_IPTTS_all,
        )
        IPTTS_valid_dataset = Dataset(
            encodings=X_IPTTS_valid_tokenized,
            labels=y_IPTTS_valid,
        )

        idxs_male_valid = [
            i
            for i in range(len(IPTTS_valid["gender"].values))
            if IPTTS_valid["gender"].values[i] == "male"
        ]
        idxs_female_valid = [
            i
            for i in range(len(IPTTS_valid["gender"].values))
            if IPTTS_valid["gender"].values[i] == "female"
        ]


        return (
            train_dataset,
            val_dataset,
            test_dataset,
            IPTTS_all_dataset,
            IPTTS_valid_dataset,
            idxs_male_valid,
            idxs_female_valid
        )

    return train_dataset, val_dataset, test_dataset
