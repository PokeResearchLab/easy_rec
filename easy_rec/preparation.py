from . import data_generation_utils, rec_torch
from copy import deepcopy
import numpy as np

def prepare_rec_data(cfg, data_params=None):
    if data_params is None:
        data_params = deepcopy(cfg["data_params"])

    data, maps = data_generation_utils.preprocess_dataset(**data_params)

    return data, maps

def prepare_rec_dataloaders(cfg, data, maps=None, data_params=None, collator_params=None, loader_params=None):
    if data_params is None:
        data_params = deepcopy(cfg["data_params"])

    datasets = rec_torch.prepare_rec_datasets(data,**data_params["dataset_params"])

    if collator_params is None:
        collator_params = deepcopy(cfg["data_params"]["collator_params"])
    if "num_items" not in collator_params:
        if maps is None:
            raise ValueError("Item mapping must be provided if num_items is not in collator_params")
        collator_params["num_items"] = data_generation_utils.get_max_number_of(maps, "sid")
    collators = rec_torch.prepare_rec_collators(**collator_params)

    if loader_params is None:
        loader_params = deepcopy(cfg["model"]["loader_params"])
    loaders = rec_torch.prepare_rec_data_loaders(datasets, **loader_params, collate_fn=collators)

    return loaders


def prepare_rec_model(cfg, maps=None, data_params=None, rec_model_params=None):
    if data_params is None:
        data_params = deepcopy(cfg["data_params"])
    
    if rec_model_params is None:
        rec_model_params = deepcopy(cfg["model"]["rec_model"])
    if "num_items" not in rec_model_params:
        if maps is None:
            raise ValueError("Item mapping must be provided if num_items is not in rec_model_params")
        rec_model_params["num_items"] = data_generation_utils.get_max_number_of(maps, "sid")
    if "num_users" not in rec_model_params:
        if maps is None:
            raise ValueError("User mapping must be provided if num_users is not in rec_model_params")
        rec_model_params["num_users"] = data_generation_utils.get_max_number_of(maps, "uid")
    if "lookback" not in rec_model_params:
        rec_model_params["lookback"] = data_params["collator_params"]["lookback"]

    main_module = rec_torch.create_rec_model(**rec_model_params)#, graph=easy_rec.data_generation_utils.get_graph_representation(data["train_sid"]))
    
    return main_module