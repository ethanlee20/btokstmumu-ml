
from pathlib import Path

from pandas import read_parquet

from ..constants import Names_of_Variables


def make_path_to_bkg_file(
    path_to_aggregated_generic_data_dir,
    charge_or_mix,
    split,
):
    
    assert charge_or_mix in ("charge", "mix")

    filename = f"{charge_or_mix}_sb_bkg_{split}.parquet"
    path = Path(path_to_aggregated_generic_data_dir).joinpath(filename)
    return path


def load_raw_background_file(
    path_to_aggregated_generic_data_dir,
    charge_or_mix,
    split,
    dtype="float32",
    columns=Names_of_Variables().list_,
    verbose=True,    
):
    
    file_path = make_path_to_bkg_file(
        path_to_aggregated_generic_data_dir=path_to_aggregated_generic_data_dir,
        charge_or_mix=charge_or_mix,
        split=split
    )

    dataframe = read_parquet(file_path)[columns]
    dataframe = dataframe.astype(dtype)

    if verbose:
        print(f"Loaded background file: {file_path} ", f"dtype: {dtype}")   
    return dataframe


def load_charge_and_mix_background_dataframes(
    path_to_aggregated_generic_data_dir,
    split,
    verbose=True
):
    
    charge_dataframe = load_raw_background_file(
        path_to_aggregated_generic_data_dir=path_to_aggregated_generic_data_dir,
        charge_or_mix="charge",
        split=split,
        verbose=verbose
    )
    mix_dataframe = load_raw_background_file(
        path_to_aggregated_generic_data_dir=path_to_aggregated_generic_data_dir,
        charge_or_mix="mix",
        split=split,
        verbose=verbose
    )
    return charge_dataframe, mix_dataframe