import torch
import pytorch_lightning as pl
import multiprocessing
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset, TensorDataset #TODO: remove TensorDataset
from . import model

#TODO: pass this function inside torch_utils
# Define a custom PyTorch Dataset class named DictDataset
class DictDataset(Dataset):
    """
    Custom PyTorch Dataset class that takes a dictionary as input and returns items based on keys.

    Args:
        data (dict): Input dictionary containing data.

    Returns:
        dict: A dictionary where each key corresponds to a tensor item.
    """
    # Constructor to initialize the dataset with input data
    def __init__(self, data):
        self.data = data

        # Convert each value in the data dictionary to a PyTorch tensor
        for key, value in self.data.items():
            if isinstance(value, torch.Tensor):
                self.data[key] = value.clone().detach()
            else:
                self.data[key] = torch.tensor(value)

    # Method to get an item from the dataset at a given index
    def __getitem__(self, index):
        return {key: value[index] for key, value in self.data.items()}

    # Method to get the length of the dataset
    def __len__(self):
        # Assumes that all values in the data dictionary have the same length
        return len(self.data[list(self.data.keys())[0]])

#TODO: pass this function inside torch_utils
# Define a custom PyTorch Dataset class named DictSequentialDataset that extends DictDataset
class DictSequentialDataset(DictDataset):
    # Constructor to initialize the sequential dataset with additional parameters
    """
    Custom PyTorch Dataset class for sequential data, extending the DictDataset.

    Args:
        data (dict): Input dictionary containing sequential data.
        sequential_keys (list): List of keys in the data dictionary representing sequential data.
        padding_value (float): Value to use for padding sequences.
        left_pad (bool): If True, pad sequences on the left; otherwise, pad on the right.
        lookback (int): Number of time steps to look back in the sequences. (length of input sequence)
        stride (int): Stride for selecting subsequences.
        lookforward (int): Number of time steps to look forward in the sequences.
        simultaneous_lookforward (int): Number of simultaneous time steps to look forward.
        out_seq_len (int): Length of the output sequence.
        keep_last (int): Number of samples to keep in the output sequence.
        drop_original (bool): If True, drop the original keys from the data dictionary.

    Returns:
        dict: A dictionary containing input and output sequences for each key.
    """
    def __init__(self, 
                 data, 
                 sequential_keys=None, 
                 padding_value=0, 
                 left_pad=True, 
                 lookback=None, 
                 stride=None, 
                 lookforward=1, 
                 simultaneous_lookforward=1,
                 out_seq_len=None,
                 keep_last = None,
                 drop_original=True):
        
        self.data = data

        # If sequential_keys is not provided, use all keys in the data dictionary
        if sequential_keys is None:
            print("WARNING: sequential_keys not provided. Using all keys in the data dictionary.")
            sequential_keys = list(data.keys())

        # If lookback and stride are not provided, set them based on the maximum length of values in the data
        if lookback is None:
            lookback = max([value.shape[-1] for value in data.values()]) + lookforward + simultaneous_lookforward
        if stride is None:
            stride = lookback
        if keep_last is None:
            keep_last = lookback

        # Pad the sequences in the data using specified parameters
        for key in sequential_keys:
            if left_pad:
                x_function = lambda x: x[::-1]
                out_func = lambda x: x.flip(dims=[1])
            else:
                x_function = lambda x: x
                out_func = lambda x: x

            needed_length = lookback + lookforward + simultaneous_lookforward
            extra_pad = lambda x : x if needed_length <= x.shape[1] else torch.cat([x, torch.zeros((x.shape[0], needed_length - x.shape[1]),dtype=x.dtype)],dim=1)

            self.data[key] = out_func(extra_pad(self.pad_list_of_tensors(self.data[key], padding_value=padding_value, x_function=x_function)))

        # Pair input and output sequences based on specified parameters
        self.pair_input_output(sequential_keys, padding_value, lookback, stride, lookforward, simultaneous_lookforward, out_seq_len, keep_last, drop_original)

        # Call the constructor of the parent class (DictDataset)
        super().__init__(self.data)

    # Method to pad a list of tensors and return the padded sequence as a tensor
    def pad_list_of_tensors(self, list_of_tensors, padding_value=0, x_function= lambda x: x):
        padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(x_function(x)) for x in list_of_tensors], batch_first=True, padding_value=padding_value)
        # Change type to type of first non-empy list in list of tensors
        for x in list_of_tensors:
            if len(x)>0: break
        padded = padded.type(getattr(torch,str(type(x[0]).__name__)))
        return padded
    # Method to pair input and output sequences based on specified parameters
    def pair_input_output(self, sequential_keys, padding_value, lookback, stride, lookforward, simultaneous_lookforward, out_seq_len, keep_last, drop_original=True):
        key_to_use = sequential_keys[0]
        max_len = self.data[key_to_use].shape[1]
        if out_seq_len is None: out_seq_len = max_len

        # Calculate input and output indices based on lookback, stride, and lookforward
        # input_indices = torch.stack([torch.arange(a-lookback,a) for a in range(max_len-lookforward, lookback-1, -stride)][::-1])
        input_indices = torch.stack([torch.arange(a-lookback,a) for a in range(max_len-lookforward-simultaneous_lookforward+1, max(lookback-1, max_len-lookforward-simultaneous_lookforward+1-out_seq_len), -stride)][::-1])
        # output_indices = torch.stack([torch.arange(a-lookback,a) for a in range(max_len, lookback-1+lookforward, -stride)][::-1])
        output_indices = torch.stack([torch.stack([torch.arange(b-simultaneous_lookforward+1,b+1) for b in torch.arange(a-lookback,a)]) for a in range(max_len, max(lookback-1+lookforward+simultaneous_lookforward-1,max_len-out_seq_len), -stride)][::-1])
        
        # Get non-sequential keys in the data dictionary
        non_sequential_keys = [key for key in self.data.keys() if key not in sequential_keys]

        # Process each sequential key
        for key in sequential_keys:
            # Create input and output sequences based on calculated indices
            self.data[f"in_{key}"] = self.data[key][:,input_indices]
            self.data[f"out_{key}"] = self.data[key][:,output_indices]

            # Remove output values where input is padding
            input_is_padding = torch.isclose(self.data[f"in_{key}"], padding_value*torch.ones_like(self.data[f"in_{key}"]))
            self.data[f"out_{key}"][input_is_padding] = padding_value

            # Remove rows where all input or all output is padding
            to_keep = torch.logical_and(
                torch.logical_not(input_is_padding.all(-1)),
                torch.logical_not(torch.isclose(self.data[f"out_{key}"], padding_value*torch.ones_like(self.data[f"out_{key}"])).all(-1).all(-1)))

            self.data[f"in_{key}"] = self.data[f"in_{key}"][to_keep]
            self.data[f"out_{key}"] = self.data[f"out_{key}"][to_keep]

            # Remove output values if index is before out_seq_len from the end
            # Option 1: keep same shape
            # self.data[f"out_{key}"][:, :-out_seq_len] = padding_value
            # Option 2: shorten array
            self.data[f"out_{key}"] = self.data[f"out_{key}"][:, max(-keep_last,-out_seq_len+self.data[f"out_{key}"].shape[-1]-1):]
            # Shorten by number of samples reserved to this split, also removing simultaneous_lookforward

            # Optional: Squeeze out the last dimension if simultaneous_lookforward is 1
            # if simultaneous_lookforward == 1:
            #     self.data[f"out_{key}"] = self.data[f"out_{key}"].squeeze(-1)

            # Optionally, drop the original key from the data dictionary
            if drop_original:
                del self.data[key]

        # Repeat the indices of non-dropped rows for non-sequential keys
        orig_rows_repeat = torch.where(to_keep)[0]

        # Process each non-sequential key
        for key in non_sequential_keys:
            self.data[key] = self.data[key][orig_rows_repeat]

# Define a custom PyTorch DataLoader class for recommendation datasets, inheriting from DataLoader
class RecommendationDataloader(DataLoader):
    # Constructor to initialize the dataloader with required parameters and additional options
    """
    Custom PyTorch DataLoader class for recommendation datasets, extending the base DataLoader class.

    Args:
        dataset (Dataset): PyTorch Dataset object containing recommendation data.
        original_sequences (list): List of original sequences.
        num_items (int): Number of items in the dataset.
        primary_key (str): Primary key for the dataset.
        relevance (str): Type of relevance function to use.
        num_negatives (int): Number of negative samples to generate.
        padding_value (float): Value used for padding sequences.
        mask_value (float): Value used for masking sequences.
        mask_prob (float): Probability of applying masking to input sequences.
        **kwargs: Additional keyword arguments for DataLoader.

    Returns:
        None
    """
    def __init__(self, 
                 dataset, 
                 original_sequences, 
                 num_items, 
                 primary_key="sid", 
                 relevance=None, 
                 num_negatives=1,
                 padding_value=0,
                 mask_value = None,
                 mask_prob = 0,
                 **kwargs):
        # Call the constructor of the parent class (DataLoader)
        super().__init__(dataset, **kwargs)

        # Initialize basic parameters
        self.num_items = num_items
        # Convert original_sequences to a list of sets for faster lookup
        self.original_sequences = [set(x) for x in original_sequences]
        self.in_key = f"in_{primary_key}"
        self.out_key = f"out_{primary_key}"

        # Get the shape of the output from the dataset
        for app in dataset:
            break
        # Set the number of positives based on the shape of the output
        self.num_positives = app[self.out_key].shape[-1]

        self.num_negatives = num_negatives

        # Generate a relevance function based on the specified relevance type and shape of the output
        self.relevance_function = generate_relevance(relevance, self.out_key)

        self.padding_value = padding_value

        if mask_prob > 0 and mask_value is None:
            mask_value = num_items
        self.mask_value = mask_value
        self.mask_prob = mask_prob

    # Method to sample negative items for a given set of indices
    def sample_negatives(self, indices, t=1):
        if self.num_negatives == 0:
            return torch.zeros(len(indices), t, self.num_negatives, dtype=torch.long)
        negatives = torch.zeros(len(indices), self.num_negatives*t, dtype=torch.long)
        for i, index in enumerate(indices):
            # Get possible negative items that are not in the original sequence
            possible_negatives = torch.tensor(list(set(range(1, self.num_items + 1)).difference(self.original_sequences[index-1]))) #-1 is needed because index starts from 1
            # Randomly sample num_negatives negative items
            negatives[i] = possible_negatives[torch.randint(0, len(possible_negatives), (self.num_negatives*t,))]
            #negatives[i] = possible_negatives[torch.randperm(len(possible_negatives))[:self.num_negatives*t]]
        negatives = negatives.reshape(len(indices), t, max(self.num_negatives,1))
        return negatives
    
    def mask_input(self, input):
        if self.mask_prob == 0: return input
        mask = torch.rand(input.shape) < self.mask_prob
        mask[:,-1] = True #Always mask last item (for val / test purposes) #TODO: generalize to more testing items
        input[mask] = self.mask_value
        return input
    # TODO: deleting objects?

    # Custom iterator method to yield batches with additional information
    def __iter__(self):
        for out in super().__iter__():
            # Add negative samples and relevance scores to the batch
            out["relevance"] = self.relevance_function(out)
            timesteps = out[self.out_key].shape[1]
            negatives = self.sample_negatives(out["uid"], timesteps)
            out[self.in_key] = self.mask_input(out[self.in_key])
            relevant_in = out[self.in_key][:, -out[self.out_key].shape[1]:]
            not_to_use = torch.isclose(out[self.out_key], self.padding_value*torch.ones_like(out[self.out_key])).all(-1) #.all(-1) for out because out can have multiple values, i.e. simultaneous_lookforward
            if self.mask_value is not None:
                not_to_use = torch.logical_or(not_to_use, torch.logical_not(torch.isclose(relevant_in, self.mask_value*torch.ones_like(relevant_in))))
            #negatives[out_is_padding] = self.padding_value # Pad negative if out (i.e. positives) is padding
            
            # Masked tensor are a prototype in torch, but could solve many issues with padding and not wanting to compute losses or metrics on padding
            #negatives = torch.masked.masked_tensor(negatives, out_is_padding.unsqueeze(-1))

            #concatenate negatives to out[out_key]
            out[self.out_key] = torch.cat([out[self.out_key], negatives], dim=-1)
            #concatenate 0 relevance to relevance
            out["relevance"] = torch.cat([out["relevance"], torch.zeros_like(negatives)], dim=-1)

            out["relevance"][not_to_use] = float("nan") # Nan relevance if out is padding

            # Yield the modified batch
            yield out

# TODO? Put inside RecommendationDataloader
            # TODO: delete this functions beacuse dont use them in our code?
# Function to generate a relevance function based on the specified relevance type and data shape
def generate_relevance(relevance, out_key):
    if relevance in {None, "fixed", "linear", "exponential"}:
        # Generate a relevance tensor based on the specified type and data shape
        # Define a lambda function to return the generated relevance tensor
        relevance_function = lambda data: generate_relevance_from_type(relevance, data[out_key])
    elif isinstance(relevance, str):
        # Use an existing output key as the relevance function
        relevance_function = lambda data: data[f"out_{relevance}"]
    else:
        # Raise an error for unsupported relevance types
        raise NotImplementedError(f"Unsupported relevance: {relevance}")

    return relevance_function

# TODO? Put inside RecommendationDataloader
# Function to generate a relevance tensor based on the specified relevance type and shape
def generate_relevance_from_type(relevance_type, data):
    shape = data.shape
    if relevance_type is None or relevance_type == "fixed":
        # Generate a tensor of ones if the relevance type is None or fixed
        app = torch.ones(shape[-1])
    else:
        # Generate a tensor with values from 1 to 0 with equal spacing
        app = torch.linspace(0, 1, shape[-1])[::-1]
        # Adjust the tensor based on the relevance type
        if relevance_type == "linear":
            pass
        elif relevance_type == "exponential":
            app = torch.exp(app)

    # Normalize the tensor
    app /= torch.sum(app)
    # Repeat the tensor to match the shape
    app = app.repeat(*shape[:-1], 1)
    return app


def prepare_rec_datasets(data,
                         split_keys={"train": ["sid", "timestamp", "rating", "uid"],
                                     "val": ["sid", "timestamp", "rating", "uid"],
                                     "test": ["sid", "timestamp", "rating", "uid"]},
                         **dataset_params
                         ):
    """
    Prepare recommendation datasets for training and evaluation.

    Args:
        data (dict): Input dictionary containing data.
        split_keys (dict): Dictionary specifying keys for different splits.
        **dataset_params: Additional parameters for dataset preparation.

    Returns:
        dict: Dictionary containing prepared recommendation datasets for each split.
    """

    datasets = {}
    for split_name, data_keys in split_keys.items():
        split_dataset_params = deepcopy(dataset_params)
        # Select specific parameters for this split
        for key, value in split_dataset_params.items():
            if isinstance(value, dict):
                if split_name in value.keys():
                    split_dataset_params[key] = value[split_name]
        
        # Get data
        data_to_use = {}
        for key in data_keys:
            if key in data:
                data_to_use[key] = data[key]
            else: #If key is not in data, try to get it using split_name
                data_to_use[key] = data[f"{split_name}_{key}"]

        # Create the DataLoader
        datasets[split_name] = DictSequentialDataset(data_to_use, **split_dataset_params)

    return datasets

def prepare_rec_data_loaders(datasets, data,
                             split_keys = ["train", "val", "test"],
                             original_seq_key="sid",
                             **loader_params):
    """
    Prepare recommendation data loaders for training and evaluation.

    Args:
        datasets (dict): Dictionary containing prepared recommendation datasets.
        data (dict): Input dictionary containing data.
        split_keys (list): List of split keys for data loaders.
        original_seq_key (str): Key for original sequences in the data.
        **loader_params: Additional parameters for data loader preparation.

    Returns:
        dict: Dictionary containing prepared recommendation data loaders for each split.
    """                         
    # TODO: dict instead of list
    # I don't remember what I meant by this comment...
    
    # Default loader parameters
    default_loader_params = {
        "num_workers": multiprocessing.cpu_count(),
        "pin_memory": True,
        "persistent_workers": True,
        "drop_last": {"train": False, "val": False, "test": False}, #TODO: check specifics about drop last: losing data?
        "shuffle": {"train": True, "val": False, "test": False},
    }
    # Combine default and custom loader parameters
    loader_params = dict(list(default_loader_params.items()) + list(loader_params.items()))
    
    loaders = {}
    for split_name in split_keys:
        split_loader_params = deepcopy(loader_params)
        # Select specific parameters for this split
        for key, value in split_loader_params.items():
            if isinstance(value, dict):
                if split_name in value.keys():
                    split_loader_params[key] = value[split_name]

        if original_seq_key in data:
            original_seq = data[original_seq_key]
        else:
            original_seq = data[f"{split_name}_{original_seq_key}"]
        
        # Create the DataLoader
        loaders[split_name] = RecommendationDataloader(datasets[split_name], original_seq, **split_loader_params)

    return loaders

def create_rec_model(name, seed=42, **model_params):
    """
    Create a recommendation model.

    Args:
        name (str): Name of the recommendation model.
        seed (int): Random seed for weight initialization.
        **model_params: Additional parameters for model creation.

    Returns:
        torch.nn.Module: Instance of the recommendation model.
    """
    # Set a random seed for weight initialization
    pl.seed_everything(seed)
    # Get the model from the model module
    return getattr(model, name)(**model_params)
