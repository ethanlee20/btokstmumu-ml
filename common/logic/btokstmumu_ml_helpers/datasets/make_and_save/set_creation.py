
import torch

from ..constants import Names_of_Variables, Names_of_Labels
from .aggregated_background import load_charge_and_mix_background_dataframes
from .aggregated_signal import Aggregated_Signal_Dataframe_Handler
from .tensor_conversion import torch_tensor_from_pandas
from .preprocessing import apply_signal_preprocessing, apply_background_preprocessing


def _bootstrap_labeled_sets(
    features_tensor, 
    labels_tensor, 
    num_events_per_set, 
    num_sets_per_label, 
    reduce_labels=True, 
):  

    labels_to_sample = torch.unique(
        labels_tensor, 
        sorted=True
    )
    tensor_of_feature_sets = torch.zeros(
        len(labels_to_sample)*num_sets_per_label, 
        num_events_per_set, 
        len(features_tensor[0]), 
        dtype=torch.float32
    )
    tensor_of_label_sets = torch.zeros(
        len(labels_to_sample)*num_sets_per_label,
        num_events_per_set,
        dtype=torch.float32
    )

    index = 0
    for label in labels_to_sample:
        for _ in range(num_sets_per_label):
            source_features = features_tensor[labels_tensor==label]
            source_labels = labels_tensor[labels_tensor==label]
            assert source_features.shape[0] == source_labels.shape[0]
            selection_indices = torch.randint(
                low=0, 
                high=len(source_features), 
                size=(num_events_per_set,)
            )
            bootstrapped_feature_set = source_features[selection_indices]
            bootstrapped_label_set = source_labels[selection_indices]
            tensor_of_feature_sets[index] = bootstrapped_feature_set
            tensor_of_label_sets[index] = bootstrapped_label_set
            index += 1

    if reduce_labels:
        tensor_of_label_sets = (
            torch.unique_consecutive(tensor_of_label_sets, dim=1).squeeze()
        )
        assert (
            tensor_of_label_sets.shape[0] 
            == tensor_of_feature_sets.shape[0]
        )

    return tensor_of_feature_sets, tensor_of_label_sets


def make_unbinned_labeled_signal_sets(
    settings, 
    list_of_variables_to_standard_scale,
    verbose=True
):
    
    aggregated_signal_dataframe_handler = Aggregated_Signal_Dataframe_Handler(
        path_to_common_processed_datasets_dir=settings.common.path_to_common_processed_datasets_dir, 
        level=settings.common.level, 
        trial_range=settings.common.raw_signal_trial_range, 
    )
    aggregated_signal_dataframe = aggregated_signal_dataframe_handler.get_dataframe()

    aggregated_signal_dataframe = apply_signal_preprocessing(
        dataframe=aggregated_signal_dataframe, 
        settings=settings,
        list_of_variables_to_standard_scale=list_of_variables_to_standard_scale,
        verbose=verbose
    )

    labels_source_tensor = torch_tensor_from_pandas(
        aggregated_signal_dataframe[Names_of_Labels().unbinned]
    )
    features_source_tensor = torch_tensor_from_pandas(
        aggregated_signal_dataframe[Names_of_Variables().list_]
    )

    feature_sets, label_sets = (
        _bootstrap_labeled_sets(
            features_tensor=features_source_tensor,
            labels_tensor=labels_source_tensor,
            num_events_per_set=settings.set.num_signal_events_per_set,
            num_sets_per_label=settings.set.num_sets_per_label,
            reduce_labels=True,
        )
    )
    return feature_sets, label_sets


def make_binned_labeled_signal_sets(settings, verbose=True):

    aggregated_signal_dataframe_handler = Aggregated_Signal_Dataframe_Handler(
        path_to_common_processed_datasets_dir=settings.common.path_to_processed_datasets_dir, 
        level=settings.common.level, 
        trial_range=settings.common.raw_signal_trial_range, 
    )
    aggregated_signal_dataframe = aggregated_signal_dataframe_handler.get_dataframe()
    bin_map = aggregated_signal_dataframe_handler.get_bin_map()
    
    aggregated_signal_dataframe = apply_signal_preprocessing(
        dataframe=aggregated_signal_dataframe, 
        settings=settings,
        list_of_variables_to_standard_scale=Names_of_Variables().list_,
        verbose=verbose
    )

    labels_source_tensor = torch_tensor_from_pandas(
        aggregated_signal_dataframe[Names_of_Labels().binned]
    )
    features_source_tensor = torch_tensor_from_pandas(
        aggregated_signal_dataframe[Names_of_Variables().list_]
    )

    feature_sets, label_sets = (
        _bootstrap_labeled_sets(
            features_tensor=features_source_tensor,
            labels_tensor=labels_source_tensor,
            num_events_per_set=settings.set.num_events_per_set_signal,
            num_sets_per_label=settings.set.num_sets_per_label,
            reduce_labels=True,
        )
    )

    return feature_sets, label_sets, bin_map


def _bootstrap_background_sets(
    charge_background_dataframe,
    mix_background_dataframe,
    num_events_per_set,
    num_sets,
    charge_fraction,
):
    
    assert (charge_fraction <= 1) and (charge_fraction >= 0)
    
    list_of_sets = []
    num_of_charge_events_per_set = int(num_events_per_set * charge_fraction)
    num_of_mix_events_per_set = num_events_per_set - num_of_charge_events_per_set
    for _ in range(num_sets):
        mix_set_tensor = torch_tensor_from_pandas(
            mix_background_dataframe.sample(
                n=num_of_mix_events_per_set, 
                replace=True
            )
        )
        charge_set_tensor = torch_tensor_from_pandas(
            charge_background_dataframe.sample(
                n=num_of_charge_events_per_set, 
                replace=True
            )
        )
        mix_and_charge_set_tensor = torch.concat(
            [mix_set_tensor, charge_set_tensor]
        )
        list_of_sets.append(
            torch.unsqueeze(mix_and_charge_set_tensor, dim=0)
        )
    
    tensor_of_sets = torch.concat(list_of_sets)
    return tensor_of_sets


def make_background_sets(
    settings,
    num_sets,
    verbose=True
):
  
    charge_background_dataframe, mix_background_dataframe = load_charge_and_mix_background_dataframes(
        path_to_aggregated_generic_data_dir=settings.common.path_to_common_processed_datasets_dir.joinpath("aggregated_generic"),
        split=settings.common.split,
        verbose=verbose
    )

    charge_background_dataframe = apply_background_preprocessing(
        dataframe=charge_background_dataframe,
        settings=settings,
        list_of_variables_to_standard_scale=Names_of_Variables().list_,
        verbose=verbose
    )
    mix_background_dataframe = apply_background_preprocessing(
        dataframe=mix_background_dataframe,
        settings=settings,
        list_of_variables_to_standard_scale=Names_of_Variables().list_,
        verbose=verbose
    )

    background_sets = _bootstrap_background_sets(
        charge_background_dataframe=charge_background_dataframe,
        mix_background_dataframe=mix_background_dataframe,
        num_events_per_set=settings.set.num_bkg_events_per_set,
        num_sets=num_sets,
        charge_fraction=settings.set.bkg_charge_fraction
    )

    return background_sets


def shuffle_feature_and_label_tensors(feature_tensor, label_tensor):

    assert len(feature_tensor) == len(label_tensor)
    assert len(feature_tensor) > 1

    num_examples = len(label_tensor)
    indices = torch.randperm(n=num_examples)
    
    feature_tensor = feature_tensor[indices]
    label_tensor = label_tensor[indices]
    return feature_tensor, label_tensor
