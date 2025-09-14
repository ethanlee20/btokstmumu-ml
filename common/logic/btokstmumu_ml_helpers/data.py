
from pathlib import Path

import numpy
import torch
import pandas
import uproot
from scipy.stats import binned_statistic_dd


def open_tree(path_to_root_file, tree_name, verbose=True):
    
    dataframe = (
        uproot.open(f"{path_to_root_file}:{tree_name}")
        .arrays(library="pd")
    )
    if verbose:
        print(f"Opened {path_to_root_file}:{tree_name}")
    return dataframe


def open_root_file_with_one_or_more_trees(path, verbose=True):
    
    """
    Open a root file as a pandas dataframe.

    The file can contain multiple trees.
    Each tree will be labeled by a 
    pandas multi-index.
    """

    def print_message(path, tree_names): 
        def make_message(path, tree_names):
            return f"Opened {path}, containing trees: {', '.join(tree_names)}"
        message = make_message(path=path, tree_names=tree_names)
        print(message)

    file = uproot.open(path)

    tree_names = [
        name.split(';')[0] for name in ("gen", "det")
    ]
    
    list_of_dataframes = [
        file[name].arrays(library="pd") 
        for name in tree_names
    ] 
    final_dataframe = pandas.concat(list_of_dataframes, keys=tree_names)
    if verbose:
        print_message(path=path, tree_names=tree_names)
    return final_dataframe


def torch_tensor_from_pandas(dataframe):

    """
    Convert a pandas dataframe to a torch tensor.
    """

    tensor = torch.from_numpy(dataframe.to_numpy())
    return tensor


def load_background_file(
    path_to_dir,
    charge_or_mix,
    split,
    columns,
    dtype="float32",
):
    
    filename = f"{charge_or_mix}_sb_bkg_{split}.parquet"
    path = Path(path_to_dir).joinpath(filename)
    dataframe = pandas.read_parquet(path)[columns].astype(dtype)
    return dataframe


def get_info_signal_file(path):

    def to_int(string):
        assert float(int(string)) == float(string)
        return int(string)

    path = Path(path)
    assert path.is_file()

    split_name = path.name.split('_')

    info = {
        "dc9" : float(split_name[1]),
        "trial" : to_int(split_name[2])
    }
    return info


def open_signal_file(path, columns="all"):

    dataframe = pandas.read_pickle(path)

    if columns == "all":
        return dataframe
    
    return dataframe[columns]


def aggregate_signal_files(
    level, 
    trial_range, 
    path_to_dir, 
    columns="all", 
    dtype="float32",
    verbose=True
):
    
    """
    Pickle files only...
    """

    assert isinstance(trial_range, range)

    file_paths = [
        path for path in list(Path(path_to_dir).glob("*.pkl"))
        if get_info_signal_file(path)["trial"] in trial_range
    ]

    dataframes = [
        open_signal_file(path, columns)
        .loc[level]
        .assign(dc9=get_info_signal_file(path)["dc9"])
        for path in file_paths
    ]

    aggregated_dataframe = pandas.concat(dataframes).astype(dtype)

    if verbose:

        print(
            "Created aggregated signal dataframe. "
            f"Trials: {trial_range[0]} to {trial_range[-1]}"
        )

    return aggregated_dataframe


def make_agg_signal_filename(level, trial_range):

    filename = f"agg_sig_{trial_range[0]}_to_{trial_range[-1]}_{level}.parquet"
    return filename


def save_agg_signal_file(dataframe, level, trial_range, path_to_dir):

    path_to_dir = Path(path_to_dir)
    filename = make_agg_signal_filename(level, trial_range)
    path = path_to_dir.joinpath(filename)
    dataframe.to_parquet(path)


def open_agg_signal_file(level, trial_range, path_to_dir):

    path_to_dir = Path(path_to_dir)
    filename = make_agg_signal_filename(level, trial_range)
    path = path_to_dir.joinpath(filename)
    dataframe = pandas.read_parquet(path)
    return dataframe


def make_image(features, bins_per_dim, dtype="float32"):
    
    """
    Make an image like Shawn.

    Variables in the input tensor must be (in this order):             
    q^2, cosine theta mu, cosine theta K, chi

    Return an image (torch tensor). Image is binned in the
    angular variables and contains the average q^2 value per bin. 
    Empty bins are set to 0.
    
    Image has shape of (1, bins, bins, bins).
    """

    features = features.detach().numpy()

    image = torch.from_numpy(
        binned_statistic_dd(
            sample=features[:, 1:], # angular
            values=features[:, 0],  # q^2
            statistic="mean",
            bins=bins_per_dim,
            range=[(-1, 1), (-1, 1), (0, 2*numpy.pi)]
        ).statistic.astype(dtype)
    )
    image = torch.nan_to_num(image, nan=0.0)
    image = image.unsqueeze(dim=0)
    return image


def apply_q_squared_veto(dataframe, q_squared_veto):
    
    """
    Apply a q^2 veto to a dataframe of B->K*ll events.
    'tight' keeps  1 < q^2 < 8.
    'loose' keeps 0 < q^2 < 20.
    'resonances' keeps 0 < q^2 < 20 except for vetos around the resonances.
    """

    def apply_tight_cut(dataframe):
        tight_cut = (
            (dataframe["q_squared"] > 1) 
            & (dataframe["q_squared"] < 8)
        )
        return dataframe[tight_cut]
    
    def apply_loose_cut(dataframe):
        loose_cut = (
            (dataframe["q_squared"] > 0) 
            & (dataframe["q_squared"] < 20)
        )
        return dataframe[loose_cut]
    
    def apply_j_psi_resonance_cut(dataframe):
        j_psi_resonance_cut = (
            (dataframe["q_squared"] < 9)
            | (dataframe["q_squared"] > 10)
        )
        return dataframe[j_psi_resonance_cut]
    
    def apply_psi_2s_resonance_cut(dataframe):
        psi_2s_resonance_cut = (
            (dataframe["q_squared"] < 13)
            | (dataframe["q_squared"] > 14)
        )
        return dataframe[psi_2s_resonance_cut]
    
    if q_squared_veto == "tight":

        dataframe = apply_tight_cut(dataframe)
    
    elif q_squared_veto == "loose":

        dataframe = apply_loose_cut(dataframe)

    elif q_squared_veto == "resonances":

        dataframe = apply_loose_cut(dataframe)
        dataframe = apply_j_psi_resonance_cut(dataframe)
        dataframe = apply_psi_2s_resonance_cut(dataframe)

    else: raise ValueError

    dataframe = dataframe.copy()
    return dataframe


def reduce_num_per_label_to_lowest(dataframe, label_name):

    num_lowest = dataframe[label_name].value_counts().values.min()
    get_subset = lambda x : x.iloc[:num_lowest]
    dataframe = (
        dataframe.groupby(label_name, group_keys=False)[dataframe.columns]
        .apply(get_subset)
    )
    return dataframe


def bootstrap_labeled_sets(
    features, 
    labels, 
    num_events_per_set, 
    num_sets_per_label, 
):  

    unique_labels = torch.unique(labels, sorted=True)

    indices = []

    for label in unique_labels:
    
        label_indices = torch.nonzero(labels==label).T
        sampled_indices = label_indices[
            torch.randint(
                len(label_indices), 
                (num_sets_per_label, num_events_per_set)
            )
        ]
        indices.append(sampled_indices)

    indices = tuple(torch.concat(indices))

    label_sets = labels[indices]
    feature_sets = features[indices]

    set_labels = torch.unique_consecutive(label_sets, dim=1).squeeze()

    assert set_labels.shape[0] == feature_sets.shape[0]

    return feature_sets, set_labels


def bootstrap_bkg_sets(
    charge_dataframe,
    mix_dataframe,
    num_events_per_set,
    num_sets,
    charge_fraction,
):
    
    assert (charge_fraction <= 1) and (charge_fraction >= 0)
    
    num_charge_per_set = round(num_events_per_set * charge_fraction)
    num_mix_per_set = num_events_per_set - num_charge_per_set

    sets = []

    for _ in range(num_sets):

        mix_set = torch_tensor_from_pandas(
            mix_dataframe.sample(
                n=num_mix_per_set, 
                replace=True
            )
        )

        charge_set = torch_tensor_from_pandas(
            charge_dataframe.sample(
                n=num_charge_per_set, 
                replace=True
            )
        )

        set_ = torch.concat([mix_set, charge_set])

        sets.append(torch.unsqueeze(set_, dim=0))
    
    sets = torch.concat(sets)
    return sets


class Custom_Data_Loader:

    def __init__(
        self,
        features,
        labels,
        batch_size,
        shuffle,
    ):
        
        assert len(features) == len(labels)
        
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batched_indices = self.set_batched_indices()

    def set_batched_indices(self):

        self.batched_indices = self.make_batched_indices(
            num_data=len(self.labels),
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

    def __len__(self):

        return len(self.batched_indices)
    
    def __iter__(self):
        
        self.index = 0
        return self
    
    def __next__(self):

        if self.index >= len(self):

            self.set_batched_indices()
            raise StopIteration
        
        batch_indices = self.batched_indices[self.index]
        batch_features = self.features[batch_indices]
        batch_labels = self.labels[batch_indices]

        self.index += 1

        return batch_features, batch_labels
    
    def make_batched_indices(self, num_data, batch_size, shuffle):
    
        indices = torch.arange(num_data)
        if shuffle: 
            indices = indices[torch.randperm(len(indices))]

        num_batches = int(numpy.floor(num_data / batch_size))
        batched_indices = torch.reshape(
            indices[:num_batches*batch_size], 
            shape=(num_batches, batch_size)
        )
        return batched_indices


def make_unbinned_sets_dataset(
    signal_dataframe,
    num_signal_events_per_set,
    num_sets_per_label,
    bkg_signal_ratio=None,
    charge_fraction=None,
    charge_dataframe=None,
    mix_dataframe=None
):
    
    features = torch_tensor_from_pandas(
        signal_dataframe[["q_squared", "costheta_mu", "costheta_K", "chi"]]
    )
    labels = torch_tensor_from_pandas(signal_dataframe["dc9"])

    feature_sets, set_labels = bootstrap_labeled_sets(
        features, 
        labels, 
        num_signal_events_per_set, 
        num_sets_per_label, 
    )

    num_bkg_events_per_set = round(num_signal_events_per_set * bkg_signal_ratio)
    num_sets=len(set_labels)
    bkg_sets = torch.tensor([])
    if bkg_signal_ratio is not None:
        bkg_sets = bootstrap_bkg_sets(
            charge_dataframe, 
            mix_dataframe, 
            num_bkg_events_per_set, 
            num_sets, 
            charge_fraction
        )

    feature_sets = torch.cat([feature_sets, bkg_sets], dim=1)

    return feature_sets, set_labels
