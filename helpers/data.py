
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


def open_bkg_file(
    charge_or_mix,
    split,
    columns,
    dtype="float32",
):
    
    filename = f"{charge_or_mix}_sb_bkg_{split}.parquet"
    path = Path("data").joinpath(filename)
    dataframe = pandas.read_parquet(path)[columns].astype(dtype)
    return dataframe


def open_bkg_files(
    split,
    columns,
    dtype="float32"
):
    
    charge, mix = (
        open_bkg_file(kind, split, columns, dtype=dtype) 
        for kind in ["charge", "mix"]
    )
    return charge, mix


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


def to_bins(array):

    array = numpy.array(array)
    bin_map, inverse_indices = numpy.unique(array, return_inverse=True)
    bins = numpy.arange(len(bin_map))
    binned_array = bins[inverse_indices]
    return binned_array, bin_map


def aggregate_signal_files(
    level, 
    trial_range, 
    input_dir, 
    columns="all", 
    dtype="float32"
):
    
    """
    Pickle files only...
    """

    assert isinstance(trial_range, range)

    file_paths = [
        path for path in list(Path(input_dir).glob("*.pkl"))
        if get_info_signal_file(path)["trial"] in trial_range
    ]

    dataframes = [
        open_signal_file(path, columns)
        .loc[level]
        .assign(dc9=get_info_signal_file(path)["dc9"])
        for path in file_paths
    ]

    aggregated_dataframe = pandas.concat(dataframes).astype(dtype)

    binned_dc9, bin_map = to_bins(aggregated_dataframe["dc9"])

    aggregated_dataframe["dc9_bin"] = binned_dc9

    return aggregated_dataframe, bin_map


def make_agg_signal_filename(level, split):

    filename = f"agg_sig_{level}_{split}.parquet"
    return filename


def save_agg_signal_file(dataframe, level, split):

    filename = make_agg_signal_filename(level, split)
    path = Path("data").joinpath(filename)
    dataframe.to_parquet(path)


def save_bin_map(bin_map):

    path = Path("data").joinpath("bin_map.npy")
    numpy.save(path, bin_map)


def make_and_save_agg_signal_file(level, split, columns, input_dir):
    
    trials = {"train" : range(1,21), "val" : range(21, 41)}

    dataframe, bin_map = aggregate_signal_files(
        level,
        trials[split],
        input_dir,
        columns=columns
    )

    save_agg_signal_file(dataframe, level, split)
    save_bin_map(bin_map)


def open_agg_signal_file(level, split):

    filename = make_agg_signal_filename(level, split)
    path = Path("data").joinpath(filename)
    dataframe = pandas.read_parquet(path)
    return dataframe


def open_bin_map_file():

    bin_map = numpy.load(Path("data").joinpath("bin_map.npy"))
    return bin_map


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
    label_subset=None 
):  

    unique_labels = torch.unique(labels, sorted=True)
    
    if label_subset: 
        
        for label in label_subset:
            assert label in unique_labels

        unique_labels = torch.unique(label_subset, sorted=True)

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


    

def make_sets(
    level, 
    split,
    num_signal_per_set,
    num_sets_per_label,
    binned_labels=False,
    label_subset=None,
    bkg_signal_ratio=None,
    charge_bkg_fraction=None,
):
    
    feature_names = ["q_squared", "costheta_mu", "costheta_K", "chi"]
    
    signal_level = "det" if level == "det_bkg" else level
    signal_dataframe = open_agg_signal_file(signal_level, split)
    
    features = torch_tensor_from_pandas(signal_dataframe[feature_names])
    
    label_name = "dc9_bin" if binned_labels else "dc9"
    labels = torch_tensor_from_pandas(signal_dataframe[label_name])

    sets_features, sets_labels = bootstrap_labeled_sets(
        features, 
        labels, 
        num_signal_per_set, 
        num_sets_per_label, 
        label_subset=label_subset
    )

    if level == "det_bkg":

        charge_bkg_df, mix_bkg_df = open_bkg_files(split, feature_names)

        num_bkg_per_set = round(num_signal_per_set * bkg_signal_ratio)
        num_sets = len(sets_labels)
        bkg_sets = bootstrap_bkg_sets(
            charge_bkg_df, 
            mix_bkg_df, 
            num_bkg_per_set, 
            num_sets, 
            charge_bkg_fraction
        )

        sets_features = torch.cat([sets_features, bkg_sets], dim=1)

    return sets_features, sets_labels


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


def make_images(
    level,
    split,
    num_signal_per_image,
    num_images_per_label,
    bins_per_dim,
    label_subset=None,
    bkg_signal_ratio=None,
    charge_bkg_fraction=None,
    dtype="float32",
):
    
    sets_features = make_sets(
        level,
        split,
        num_signal_per_image,
        num_images_per_label,
        label_subset=label_subset,
        bkg_signal_ratio=bkg_signal_ratio,
        charge_bkg_fraction=charge_bkg_fraction
    )
    
    images = [
        make_image(set_, bins_per_dim, dtype=dtype)
        .unsqueeze(dim=0) 
        for set_ in sets_features
    ]

    return torch.cat(images)


def make_events(level, split, binned_labels=True):

    if not binned_labels: raise NotImplementedError
    
    feature_names = ["q_squared", "costheta_mu", "costheta_K", "chi"]
    label_name = "dc9_bin"

    signal_dataframe = open_agg_signal_file(level, split)

    features = torch_tensor_from_pandas(signal_dataframe[feature_names])
    labels = torch_tensor_from_pandas(signal_dataframe[label_name])

    return features, labels


def make_dset_file_path(dset_name, level, split, kind, num_signal_per_set=None): 

    filename = (
        f"{dset_name}_{level}_{split}_{num_signal_per_set}_{kind}.pt" 
        if num_signal_per_set else f"{dset_name}_{level}_{split}_{kind}.pt"
    )

    return Path("data").joinpath(filename) 


def save_dset_file(data, dset_name, level, split, kind, num_signal_per_set=None):
    
    path = make_dset_file_path(dset_name, level, split, kind, num_signal_per_set=num_signal_per_set)
    
    torch.save(data, path)


def open_dset_file(dset_name, level, split, kind, num_signal_per_set=None):

    path = make_dset_file_path(dset_name, level, split, kind, num_signal_per_set=num_signal_per_set)

    return torch.load(path, weights_only=True)


def apply_std_scale(features, dset_name, level, num_signal_per_set=None):

    std_scale_mean = open_dset_file(dset_name, level, "train", "mean", num_signal_per_set=num_signal_per_set)
    std_scale_std = open_dset_file(dset_name, level, "train", "std", num_signal_per_set=num_signal_per_set)

    return (features - std_scale_mean) / std_scale_std


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dset_name, level, split, num_signal_per_set=None, sensitivity=False):

        features_name = "sens_features" if sensitivity else "features"
        labels_name = "sens_labels" if sensitivity else "labels"

        features = open_dset_file(dset_name, level, split, features_name, num_signal_per_set=num_signal_per_set)
        labels = open_dset_file(dset_name, level, split, labels_name, num_signal_per_set=num_signal_per_set)

        assert len(features) == len(labels)

        self.features = features
        self.labels = labels

    def __len__(self): 

        return len(self.labels)


class Data_Loader:

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
    ):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.set_batched_indices()

    def set_batched_indices(self):

        indices = torch.arange(len(self.dataset))
        if self.shuffle: 
            indices = indices[torch.randperm(len(indices))]

        num_batches = int(numpy.floor(len(self.dataset) / self.batch_size))
        self.batched_indices = torch.reshape(
            indices[:num_batches*self.batch_size], 
            shape=(num_batches, self.batch_size)
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
        batch_features = self.dataset.features[batch_indices]
        batch_labels = self.dataset.labels[batch_indices]

        self.index += 1

        return batch_features, batch_labels