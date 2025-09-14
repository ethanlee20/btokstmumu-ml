
from collections import namedtuple
from itertools import product

import matplotlib.pyplot as plt

from ..datasets.settings.settings import Unbinned_Sets_Dataset_Settings, Images_Dataset_Settings, Binned_Events_Dataset_Settings, Binned_Sets_Dataset_Settings
from ..datasets.constants import Names_of_Splits, Numbers_of_Signal_Events_per_Set, Names_of_Levels
from ..datasets.datasets import Unbinned_Sets_Dataset, Images_Dataset, Binned_Events_Dataset, Binned_Sets_Dataset
from ..models.settings import Model_Settings
from ..models.constants import Names_of_Models
from ..models.models import Model
from ..models.trainer import Trainer
from ..models.evaluation import Set_Based_Model_Evaluator, Event_Based_Model_Evaluator
from .constants import Paths_to_Directories, delta_C9_value_new_physics, delta_C9_value_standard_model, Names_of_Result_Table_Columns
from ..plot.linearity import plot_linearity
from ..plot.sensitivity import plot_sensitivity
from ..plot.probabilities import plot_log_probability_distribution_examples
from ..plot.image import plot_image_examples
from .results_table import Results_Table


def to_dict_over_num_signal_events_per_set(x):

    def to_dict(x):
        return {
            num_events_per_set : x
            for num_events_per_set in Numbers_of_Signal_Events_per_Set().tuple_
        }
    
    if type(x) is int:
        x = to_dict(x)
    if type(x) is not dict: 
        raise ValueError
    if sorted(tuple(x.keys())) != sorted(Numbers_of_Signal_Events_per_Set().tuple_):
        raise ValueError
    return x


def train_model(
    model,
    training_dataset,
    evaluation_dataset,
    device      
):   
    trainer = Trainer(
        model=model,
        training_dataset=training_dataset,
        evaluation_dataset=evaluation_dataset,
        device=device
    )
    trainer.train()


def evaluate_model(
    model, 
    evaluation_dataset, 
    sensitivity_evaluation_dataset,
    results_table,
    device,
    epoch="final"
):
    
    def run_linearity_test(evaluator, evaluation_dataset):
        linearity_test_results = evaluator.run_linearity_test(evaluation_dataset)
        return linearity_test_results
    
    def run_sensitivity_test(evaluator, sensitivity_evaluation_dataset, results_table):
        sensitivity_test_results = evaluator.run_sensitivity_test(sensitivity_evaluation_dataset)
        results_table.add_items(
            column_names=[
                Names_of_Result_Table_Columns().np_mean, 
                Names_of_Result_Table_Columns().np_bias, 
                Names_of_Result_Table_Columns().np_std
            ],
            values=[
                sensitivity_test_results.avg, 
                sensitivity_test_results.bias, 
                sensitivity_test_results.std
            ],
            model_settings=evaluator.model.settings,
            dataset_settings=sensitivity_evaluation_dataset.settings,
        )
        return sensitivity_test_results
    
    def run_error_test(evaluator, evaluation_dataset, results_table):
        error_test_results = evaluator.run_error_test(evaluation_dataset)
        results_table.add_items(
            column_names=[
                Names_of_Result_Table_Columns().mse, 
                Names_of_Result_Table_Columns().mae
            ],
            values=[
                error_test_results.mse, 
                error_test_results.mae
            ],
            model_settings=evaluator.model.settings,
            dataset_settings=evaluation_dataset.settings,
        )    
        return error_test_results

    def predict_log_probability_distributions(evaluator, evaluation_dataset):
        log_probabilities = evaluator.predict_log_probabilities(evaluation_dataset.features)
        return log_probabilities

    if epoch == "final":
        model.load_final_model_from_file()
    elif type(epoch) == int:
        model.load_checkpoint_model_file(epoch)
    else: raise ValueError

    evaluator = (
        Set_Based_Model_Evaluator(model=model, device=device) 
        if model.settings.name in Names_of_Models().set_based
        else Event_Based_Model_Evaluator(model=model, device=device)
        if model.settings.name in Names_of_Models().event_based
        else None
    )
    if evaluator is None: raise ValueError

    Evaluation_Results = namedtuple(
        "Evaluation_Results", 
        ["linearity_results", "sensitivity_results", "error_results", "log_probabilities"]
    )

    results = Evaluation_Results(
        linearity_results=run_linearity_test(evaluator=evaluator, evaluation_dataset=evaluation_dataset),
        sensitivity_results=run_sensitivity_test(evaluator=evaluator, sensitivity_evaluation_dataset=sensitivity_evaluation_dataset, results_table=results_table),
        error_results=run_error_test(evaluator=evaluator, evaluation_dataset=evaluation_dataset, results_table=results_table),
        log_probabilities=(
            predict_log_probability_distributions(evaluator=evaluator, evaluation_dataset=evaluation_dataset) 
            if model.settings.name in Names_of_Models().event_based else None
        )
    )

    return results
    


class Deep_Sets:

    def __init__(
        self,
        level,
        num_events_per_set,
        num_sets_per_label,
        num_sets_per_label_sensitivity,
        q_squared_veto,
        std_scale,
        shuffle,
        uniform_label_counts,
        loss_fn,
        learning_rate,
        learning_rate_scheduler_reduction_factor,
        size_of_training_batch,
        size_of_evaluation_batch,
        number_of_epochs,
        number_of_epochs_between_checkpoints,
        results_table:Results_Table,
        device,
        bkg_fraction=None,
        bkg_charge_fraction=None
    ):  

        self.model = Model(self.model_settings)
        self.results_table = results_table
        self.device = device

        self.training_dataset_settings = Unbinned_Sets_Dataset_Settings(
            level=level,
            split=Names_of_Splits().train,
            num_events_per_set=num_events_per_set,
            num_sets_per_label=num_sets_per_label,
            is_sensitivity_study=False,
            q_squared_veto=q_squared_veto,
            std_scale=std_scale,
            shuffle=shuffle,
            uniform_label_counts=uniform_label_counts,
            path_to_processed_datasets_dir=Paths_to_Directories().path_to_main_datasets_dir,
            path_to_raw_signal_dir=Paths_to_Directories().path_to_raw_signal_dir,
            path_to_raw_bkg_dir=Paths_to_Directories().path_to_raw_bkg_dir,
            bkg_fraction=bkg_fraction,
            bkg_charge_fraction=bkg_charge_fraction,
            label_subset=None
        )

        self.evaluation_dataset_settings = Unbinned_Sets_Dataset_Settings(
            level=level,
            split=Names_of_Splits().validation,
            num_events_per_set=num_events_per_set,
            num_sets_per_label=num_sets_per_label,
            is_sensitivity_study=False,
            q_squared_veto=q_squared_veto,
            std_scale=std_scale,
            shuffle=shuffle,
            uniform_label_counts=uniform_label_counts,
            path_to_processed_datasets_dir=Paths_to_Directories().path_to_main_datasets_dir,
            path_to_raw_signal_dir=Paths_to_Directories().path_to_raw_signal_dir,
            path_to_raw_bkg_dir=Paths_to_Directories().path_to_raw_bkg_dir,
            bkg_fraction=bkg_fraction,
            bkg_charge_fraction=bkg_charge_fraction,
            label_subset=None
        )

        self.sensitivity_evaluation_dataset_settings = Unbinned_Sets_Dataset_Settings(
            level=level,
            split=Names_of_Splits().validation,
            num_events_per_set=num_events_per_set,
            num_sets_per_label=num_sets_per_label_sensitivity,
            is_sensitivity_study=True,
            q_squared_veto=q_squared_veto,
            std_scale=std_scale,
            shuffle=shuffle,
            uniform_label_counts=uniform_label_counts,
            path_to_processed_datasets_dir=Paths_to_Directories().path_to_main_datasets_dir,
            path_to_raw_signal_dir=Paths_to_Directories().path_to_raw_signal_dir,
            path_to_raw_bkg_dir=Paths_to_Directories().path_to_raw_bkg_dir,
            bkg_fraction=bkg_fraction,
            bkg_charge_fraction=bkg_charge_fraction,
            label_subset=[delta_C9_value_new_physics]
        )

        self.model_settings = Model_Settings(
            name=Names_of_Models().deep_sets,
            path_to_local_models_dir=Paths_to_Directories().path_to_main_models_dir,
            training_dataset_settings=self.training_dataset_settings,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            learning_rate_scheduler_reduction_factor=learning_rate_scheduler_reduction_factor,
            size_of_training_batch=size_of_training_batch,
            size_of_evaluation_batch=size_of_evaluation_batch,
            number_of_epochs=number_of_epochs,
            number_of_epochs_between_checkpoints=number_of_epochs_between_checkpoints
        )

    def train_model(self, remake_datasets):

        training_dataset = Unbinned_Sets_Dataset(self.training_dataset_settings)
        evaluation_dataset = Unbinned_Sets_Dataset(self.evaluation_dataset_settings)

        datasets = (training_dataset, evaluation_dataset)

        if remake_datasets:
            for dataset in datasets:
                dataset.make_and_save()

        for dataset in datasets:
            dataset.load()

        train_model(
            model=self.model, 
            training_dataset=training_dataset, 
            evaluation_dataset=evaluation_dataset, 
            device=self.device
        )

        for dataset in datasets:
            dataset.unload()
        
    def evaluate_model(self, remake_datasets, epoch="final"):
        
        evaluation_dataset = Unbinned_Sets_Dataset(self.evaluation_dataset_settings)
        sensitivity_evaluation_dataset = Unbinned_Sets_Dataset(self.sensitivity_evaluation_dataset_settings)

        datasets = (evaluation_dataset, sensitivity_evaluation_dataset)

        if remake_datasets:
            for dataset in datasets:
                dataset.make_and_save()
        
        for dataset in datasets:
            dataset.load()

        results = evaluate_model(
            model=self.model,
            evaluation_dataset=evaluation_dataset, 
            sensitivity_evaluation_dataset=sensitivity_evaluation_dataset,
            results_table=self.results_table,
            device=self.device,
            epoch=epoch
        )

        for dataset in datasets:
            dataset.unload()

        return results
        
 


class CNN:

    def __init__(
        self,
        model_variant,
        level,
        num_signal_events_per_set,
        num_sets_per_label,
        num_sets_per_label_sensitivity,
        num_bins_per_dimension,
        q_squared_veto,
        std_scale,
        shuffle,
        uniform_label_counts,
        loss_fn,
        learning_rate,
        learning_rate_scheduler_reduction_factor,
        learning_rate_scheduler_patience,
        size_of_training_batch,
        size_of_evaluation_batch,
        number_of_epochs,
        number_of_epochs_between_checkpoints,
        results_table:Results_Table,
        device,
        path_to_common_processed_datasets_dir,
        path_to_local_processed_datasets_dir,
        path_to_local_models_dir,
        bkg_fraction=None,
        bkg_charge_fraction=None
    ):
        
        assert model_variant in ("shawn", "nominal")
        
        common_dataset_settings = dict(
            level=level,
            num_signal_events_per_set=num_signal_events_per_set,
            num_bins_per_dimension=num_bins_per_dimension,
            q_squared_veto=q_squared_veto,
            std_scale=std_scale,
            shuffle=shuffle,
            uniform_label_counts=uniform_label_counts,
            bkg_fraction=bkg_fraction,
            bkg_charge_fraction=bkg_charge_fraction,
            path_to_common_processed_datasets_dir=path_to_common_processed_datasets_dir,
            path_to_local_processed_datasets_dir=path_to_local_processed_datasets_dir,
        )
        
        self.training_dataset_settings = Images_Dataset_Settings(
            **common_dataset_settings,
            split=Names_of_Splits().train,
            num_sets_per_label=num_sets_per_label,
            is_sensitivity_study=False,
            label_subset=None
        )

        self.evaluation_dataset_settings = Images_Dataset_Settings(
            **common_dataset_settings,
            split=Names_of_Splits().validation,
            num_sets_per_label=num_sets_per_label,
            is_sensitivity_study=False,
            label_subset=None
        )

        self.sensitivity_evaluation_dataset_settings = Images_Dataset_Settings(
            **common_dataset_settings,
            split=Names_of_Splits().validation,
            num_sets_per_label=num_sets_per_label_sensitivity,
            is_sensitivity_study=True,
            label_subset=[delta_C9_value_new_physics],
        )

        self.model_settings = Model_Settings(
            name=(
                Names_of_Models().cnn_shawn if model_variant=="shawn" 
                else Names_of_Models().cnn if model_variant=="nominal"
                else None
            ),
            path_to_local_models_dir=path_to_local_models_dir,
            training_dataset_settings=self.training_dataset_settings,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            learning_rate_scheduler_reduction_factor=learning_rate_scheduler_reduction_factor,
            learning_rate_scheduler_patience=learning_rate_scheduler_patience,
            size_of_training_batch=size_of_training_batch,
            size_of_evaluation_batch=size_of_evaluation_batch,
            number_of_epochs=number_of_epochs,
            number_of_epochs_between_checkpoints=number_of_epochs_between_checkpoints
        )

        self.model = Model(self.model_settings)
        self.results_table = results_table
        self.device = device

    def train_model(self, remake_datasets):
        
        training_dataset = Images_Dataset(self.training_dataset_settings)
        evaluation_dataset = Images_Dataset(self.evaluation_dataset_settings)

        datasets = (training_dataset, evaluation_dataset)

        if remake_datasets: 
            for dataset in datasets:
                dataset.make_and_save()
    
        for dataset in datasets:
            dataset.load()
        
        train_model(
            model=self.model, 
            training_dataset=training_dataset, 
            evaluation_dataset=evaluation_dataset, 
            device=self.device
        )

        for dataset in datasets:
            dataset.unload()

    def evaluate_model(self, remake_datasets, epoch="final"):
        
        evaluation_dataset = Images_Dataset(self.evaluation_dataset_settings)
        sensitivity_evaluation_dataset = Images_Dataset(self.sensitivity_evaluation_dataset_settings)

        datasets = (
            evaluation_dataset, 
            sensitivity_evaluation_dataset,
        )

        if remake_datasets:
            for dataset in datasets:
                dataset.make_and_save()

        for dataset in datasets:
            dataset.load()

        results = evaluate_model(
            model=self.model,
            evaluation_dataset=evaluation_dataset, 
            sensitivity_evaluation_dataset=sensitivity_evaluation_dataset,
            results_table=self.results_table,
            device=self.device,
            epoch=epoch
        )

        for dataset in datasets:
            dataset.unload()

        return results

    def plot_image_examples(self, remake_datasets, path_to_plots_dir):
        
        evaluation_dataset = Images_Dataset(self.evaluation_dataset_settings)

        if remake_datasets:
            evaluation_dataset.make_and_save()
        
        evaluation_dataset.load()

        plot_image_examples(
            dataset=evaluation_dataset, 
            path_to_plots_dir=path_to_plots_dir,
        )

        evaluation_dataset.unload()

        
     


class Event_by_Event:

    def __init__(
        self,
        level,
        num_evaluation_sets_per_label,
        num_evaluation_sets_per_label_sensitivity,
        q_squared_veto,
        std_scale,
        shuffle,
        uniform_label_counts,
        loss_fn,
        learning_rate,
        learning_rate_scheduler_reduction_factor,
        size_of_training_batch,
        size_of_evaluation_batch,
        number_of_epochs,
        number_of_epochs_between_checkpoints,
        results_table:Results_Table,
        device
    ):       
        
        self._initialize_settings(
            level=level,
            num_evaluation_sets_per_label=num_evaluation_sets_per_label,
            num_evaluation_sets_per_label_sensitivity=num_evaluation_sets_per_label_sensitivity,
            q_squared_veto=q_squared_veto,
            std_scale=std_scale,
            shuffle=shuffle,
            uniform_label_counts=uniform_label_counts,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            learning_rate_scheduler_reduction_factor=learning_rate_scheduler_reduction_factor,
            size_of_training_batch=size_of_training_batch,
            size_of_evaluation_batch=size_of_evaluation_batch,
            number_of_epochs=number_of_epochs,
            number_of_epochs_between_checkpoints=number_of_epochs_between_checkpoints
        )
        
        self.model = Model(self.model_settings)
        self.results_table = results_table
        self.device = device

    def train_model(self, remake_datasets):
        
        training_dataset = Binned_Events_Dataset(self.training_dataset_settings)
        evaluation_dataset = Binned_Events_Dataset(self.evaluation_event_dataset_settings)

        datasets = (training_dataset, evaluation_dataset)

        if remake_datasets:
            for dataset in datasets:
                dataset.make_and_save()

        for dataset in datasets:
            dataset.load()

        train_model(
            model=self.model, 
            training_dataset=training_dataset,
            evaluation_dataset=evaluation_dataset,
            device=self.device
        )
        
        for dataset in datasets:
            dataset.unload()

    def evaluate_model(self, num_events_per_set, remake_datasets, epoch="final"):
        
        evaluation_dataset = Binned_Sets_Dataset(
            settings=self._get_evaluation_set_dataset_settings(num_events_per_set),
        )
        sensitivity_evaluation_dataset = Binned_Sets_Dataset(
            settings=self._get_sensitivity_evaluation_set_dataset_settings(num_events_per_set),
        )

        datasets = (evaluation_dataset, sensitivity_evaluation_dataset)

        if remake_datasets:
            for dataset in datasets:
                dataset.make_and_save()

        for dataset in datasets:
            dataset.load()

        results = evaluate_model(
            model=self.model,
            evaluation_dataset=evaluation_dataset,
            sensitivity_evaluation_dataset=sensitivity_evaluation_dataset,
            results_table=self.results_table,
            device=self.device,
            epoch=epoch
        )

        for dataset in datasets:
            dataset.unload()

        return results

    def _initialize_settings(
        self,
        level,
        num_evaluation_sets_per_label,
        num_evaluation_sets_per_label_sensitivity,
        q_squared_veto,
        std_scale,
        shuffle,
        uniform_label_counts,
        loss_fn,
        learning_rate,
        learning_rate_scheduler_reduction_factor,
        size_of_training_batch,
        size_of_evaluation_batch,
        number_of_epochs,
        number_of_epochs_between_checkpoints
    ):
        
        self.training_dataset_settings = Binned_Events_Dataset_Settings(
            level=level,
            split=Names_of_Splits().train,
            q_squared_veto=q_squared_veto,
            std_scale=std_scale,
            shuffle=shuffle,
            uniform_label_counts=uniform_label_counts,
            path_to_processed_datasets_dir=Paths_to_Directories().path_to_main_datasets_dir,
            path_to_raw_signal_dir=Paths_to_Directories().path_to_raw_signal_dir,
            path_to_raw_bkg_dir=Paths_to_Directories().path_to_raw_bkg_dir
        )

        self.evaluation_event_dataset_settings = Binned_Events_Dataset_Settings(
            level=level,
            split=Names_of_Splits().validation,
            q_squared_veto=q_squared_veto,
            std_scale=std_scale,
            shuffle=shuffle,
            uniform_label_counts=uniform_label_counts,
            path_to_processed_datasets_dir=Paths_to_Directories().path_to_main_datasets_dir,
            path_to_raw_signal_dir=Paths_to_Directories().path_to_raw_signal_dir,
            path_to_raw_bkg_dir=Paths_to_Directories().path_to_raw_bkg_dir
        )

        self.evaluation_set_datasets_settings = {
            num_events_per_set : Binned_Sets_Dataset_Settings(
                level=level,
                split=Names_of_Splits().validation,
                num_events_per_set=num_events_per_set,
                num_sets_per_label=num_evaluation_sets_per_label[num_events_per_set],
                is_sensitivity_study=False,
                q_squared_veto=q_squared_veto,
                std_scale=std_scale,
                shuffle=shuffle,
                uniform_label_counts=uniform_label_counts,
                path_to_processed_datasets_dir=Paths_to_Directories().path_to_main_datasets_dir,
                path_to_raw_signal_dir=Paths_to_Directories().path_to_raw_signal_dir,
                path_to_raw_bkg_dir=Paths_to_Directories().path_to_raw_bkg_dir,
                bkg_fraction=None,
                bkg_charge_fraction=None
            )
            for num_events_per_set in Numbers_of_Signal_Events_per_Set().tuple_
        }

        self.sensitivity_evaluation_set_datasets_settings = {
            num_events_per_set : Binned_Sets_Dataset_Settings(
                level=level,
                split=Names_of_Splits().validation,
                num_events_per_set=num_events_per_set,
                num_sets_per_label=num_evaluation_sets_per_label_sensitivity,
                is_sensitivity_study=True,
                q_squared_veto=q_squared_veto,
                std_scale=std_scale,
                shuffle=shuffle,
                uniform_label_counts=uniform_label_counts,
                path_to_processed_datasets_dir=Paths_to_Directories().path_to_main_datasets_dir,
                path_to_raw_signal_dir=Paths_to_Directories().path_to_raw_signal_dir,
                path_to_raw_bkg_dir=Paths_to_Directories().path_to_raw_bkg_dir,
                bkg_fraction=None,
                bkg_charge_fraction=None,
                label_subset=[delta_C9_value_new_physics]
            )
            for num_events_per_set in Numbers_of_Signal_Events_per_Set().tuple_
        }
        
        self.model_settings = Model_Settings(
            name=Names_of_Models().ebe,
            path_to_local_models_dir=Paths_to_Directories().path_to_main_models_dir,
            training_dataset_settings=self.training_dataset_settings,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            learning_rate_scheduler_reduction_factor=learning_rate_scheduler_reduction_factor,
            size_of_training_batch=size_of_training_batch,
            size_of_evaluation_batch=size_of_evaluation_batch,
            number_of_epochs=number_of_epochs,
            number_of_epochs_between_checkpoints=number_of_epochs_between_checkpoints
        )

    def _get_evaluation_set_dataset_settings(self, num_events_per_set):
        return self.evaluation_set_datasets_settings[num_events_per_set]
    
    def _get_sensitivity_evaluation_set_dataset_settings(self, num_events_per_set):
        return self.sensitivity_evaluation_set_datasets_settings[num_events_per_set]
    


class Deep_Sets_Group:

    def __init__(
        self,
        num_sets_per_label,
        num_sets_per_label_sensitivity,
        q_squared_veto,
        std_scale,
        shuffle,
        uniform_label_counts,
        loss_fn,
        learning_rate,
        learning_rate_scheduler_reduction_factor,
        size_of_training_batch,
        size_of_evaluation_batch,
        number_of_epochs,
        number_of_epochs_between_checkpoints,
        results_table,
        device,
        bkg_fraction,
        bkg_charge_fraction
    ):      
    
        self._initialize_group(
            num_sets_per_label=num_sets_per_label,
            num_sets_per_label_sensitivity=num_sets_per_label_sensitivity,
            q_squared_veto=q_squared_veto,
            std_scale=std_scale,
            shuffle=shuffle,
            uniform_label_counts=uniform_label_counts,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            learning_rate_scheduler_reduction_factor=learning_rate_scheduler_reduction_factor,
            size_of_training_batch=size_of_training_batch,
            size_of_evaluation_batch=size_of_evaluation_batch,
            number_of_epochs=number_of_epochs,
            number_of_epochs_between_checkpoints=number_of_epochs_between_checkpoints,
            results_table=results_table,
            device=device,
            bkg_fraction=bkg_fraction,
            bkg_charge_fraction=bkg_charge_fraction
        )

    def train_all(self, remake_datasets):
        for level in Names_of_Levels().tuple_:
            for num_events_per_set in Numbers_of_Signal_Events_per_Set().tuple_:
                (
                    self.get_individual(level=level, num_events_per_set=num_events_per_set)
                    .train_model(remake_datasets=remake_datasets)
                )

    def train_subset(self, levels, nums_events_per_set, remake_datasets):
        for level in levels:
            for num_events_per_set in nums_events_per_set:
                (
                    self.get_individual(level=level, num_events_per_set=num_events_per_set)
                    .train_model(remake_datasets=remake_datasets)
                )              
        
    def evaluate_all(self, remake_datasets, epoch="final"):
        
        self.results = {
            level : {
                num_events_per_set : self.get_individual(level=level, num_events_per_set=num_events_per_set).evaluate_model(remake_datasets=remake_datasets, epoch=epoch)
                for num_events_per_set in Numbers_of_Signal_Events_per_Set().tuple_
            }
            for level in Names_of_Levels().tuple_
        }

    def evaluate_subset(self, levels, nums_events_per_set, remake_datasets, epoch="final"):
        for level in levels:
            for num_events_per_set in nums_events_per_set:
                (
                    self.get_individual(level=level, num_events_per_set=num_events_per_set)
                    .evaluate_model(remake_datasets=remake_datasets, epoch=epoch)
                )          

    def get_individual(self, level, num_events_per_set):
        return self.group[level][num_events_per_set]

    def _initialize_group(
        self,
        num_sets_per_label,
        num_sets_per_label_sensitivity,
        q_squared_veto,
        std_scale,
        shuffle,
        uniform_label_counts,
        loss_fn,
        learning_rate,
        learning_rate_scheduler_reduction_factor,
        size_of_training_batch,
        size_of_evaluation_batch,
        number_of_epochs,
        number_of_epochs_between_checkpoints,
        results_table,
        device,
        bkg_fraction,
        bkg_charge_fraction
    ):
        
        common_parameters = dict(
            num_sets_per_label_sensitivity=num_sets_per_label_sensitivity,
            q_squared_veto=q_squared_veto,
            std_scale=std_scale,
            shuffle=shuffle,
            uniform_label_counts=uniform_label_counts,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            learning_rate_scheduler_reduction_factor=learning_rate_scheduler_reduction_factor,
            number_of_epochs=number_of_epochs,
            number_of_epochs_between_checkpoints=number_of_epochs_between_checkpoints,
            results_table=results_table,
            device=device
        )

        num_sets_per_label = to_dict_over_num_signal_events_per_set(num_sets_per_label)
        size_of_training_batch = to_dict_over_num_signal_events_per_set(size_of_training_batch)
        size_of_evaluation_batch = to_dict_over_num_signal_events_per_set(size_of_evaluation_batch)

        self.group = {
            level : {
                num_events_per_set : Deep_Sets(
                    **common_parameters,
                    level=level,
                    num_events_per_set=num_events_per_set,
                    num_sets_per_label=num_sets_per_label[num_events_per_set],
                    size_of_training_batch=size_of_training_batch[num_events_per_set],
                    size_of_evaluation_batch=size_of_evaluation_batch[num_events_per_set]
                )
                for num_events_per_set in Numbers_of_Signal_Events_per_Set().tuple_
            }
            for level in (Names_of_Levels().generator, Names_of_Levels().detector)
        }

        self.group[Names_of_Levels().detector_and_background] = {
            num_events_per_set : Deep_Sets(
                **common_parameters,
                level=Names_of_Levels().detector_and_background,
                num_events_per_set=num_events_per_set,
                num_sets_per_label=num_sets_per_label[num_events_per_set],
                size_of_training_batch=size_of_training_batch[num_events_per_set],
                size_of_evaluation_batch=size_of_evaluation_batch[num_events_per_set],
                bkg_fraction=bkg_fraction,
                bkg_charge_fraction=bkg_charge_fraction
            )
            for num_events_per_set in Numbers_of_Signal_Events_per_Set().tuple_
        }


class CNN_Group:

    def __init__(
        self,
        model_variant,
        num_sets_per_label,
        num_sets_per_label_sensitivity,
        num_bins_per_dimension,
        q_squared_veto,
        std_scale,
        shuffle,
        uniform_label_counts,
        loss_fn,
        learning_rate,
        learning_rate_scheduler_reduction_factor,
        learning_rate_scheduler_patience,
        size_of_training_batch,
        size_of_evaluation_batch,
        number_of_epochs,
        number_of_epochs_between_checkpoints,
        results_table,
        device,
        bkg_fraction,
        bkg_charge_fraction,
        path_to_common_processed_datasets_dir,
        path_to_local_processed_datasets_dir,
        path_to_local_models_dir,
    ):
        
        common_parameters = dict(
            model_variant=model_variant,
            num_sets_per_label_sensitivity=num_sets_per_label_sensitivity,
            num_bins_per_dimension=num_bins_per_dimension,
            q_squared_veto=q_squared_veto,
            std_scale=std_scale,
            shuffle=shuffle,
            uniform_label_counts=uniform_label_counts,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            learning_rate_scheduler_reduction_factor=learning_rate_scheduler_reduction_factor,
            learning_rate_scheduler_patience=learning_rate_scheduler_patience,
            number_of_epochs=number_of_epochs,
            number_of_epochs_between_checkpoints=number_of_epochs_between_checkpoints,
            results_table=results_table,
            device=device,
            path_to_common_processed_datasets_dir=path_to_common_processed_datasets_dir,
            path_to_local_processed_datasets_dir=path_to_local_processed_datasets_dir,
            path_to_local_models_dir=path_to_local_models_dir,
        )

        num_sets_per_label = to_dict_over_num_signal_events_per_set(num_sets_per_label)
        size_of_training_batch = to_dict_over_num_signal_events_per_set(size_of_training_batch)
        size_of_evaluation_batch = to_dict_over_num_signal_events_per_set(size_of_evaluation_batch)

        self.group = {
            level : {
                num_signal_events_per_set : CNN(
                    **common_parameters,
                    level=level,
                    num_signal_events_per_set=num_signal_events_per_set,
                    num_sets_per_label=num_sets_per_label[num_signal_events_per_set],
                    size_of_training_batch=size_of_training_batch[num_signal_events_per_set],
                    size_of_evaluation_batch=size_of_evaluation_batch[num_signal_events_per_set]
                )
                for num_signal_events_per_set in Numbers_of_Signal_Events_per_Set().tuple_
            }
            for level in (Names_of_Levels().generator, Names_of_Levels().detector)
        }

        self.group[Names_of_Levels().detector_and_background] = {
            num_signal_events_per_set : CNN(
                **common_parameters,
                level=Names_of_Levels().detector_and_background,
                num_signal_events_per_set=num_signal_events_per_set,
                bkg_fraction=bkg_fraction,
                bkg_charge_fraction=bkg_charge_fraction,
                num_sets_per_label=num_sets_per_label[num_signal_events_per_set],
                size_of_training_batch=size_of_training_batch[num_signal_events_per_set],
                size_of_evaluation_batch=size_of_evaluation_batch[num_signal_events_per_set]
            )
            for num_signal_events_per_set in Numbers_of_Signal_Events_per_Set().tuple_
        }

        self.num_sets_per_label = num_sets_per_label
        self.num_bins_per_dimension = num_bins_per_dimension

        self.results = {
            level : {
                num_signal_events_per_set : None
                for num_signal_events_per_set in Numbers_of_Signal_Events_per_Set().tuple_
            }
            for level in Names_of_Levels().tuple_
        }

    def train_all(self, remake_datasets):
        for level in Names_of_Levels().tuple_:
            for num_signal_events_per_set in Numbers_of_Signal_Events_per_Set().tuple_:
                (
                    self.get_individual(level=level, num_signal_events_per_set=num_signal_events_per_set)
                    .train_model(remake_datasets=remake_datasets)
                )

    def train_subset(self, levels, nums_signal_events_per_set, remake_datasets):
        for level in levels:
            for num_signal_events_per_set in nums_signal_events_per_set:
                (
                    self.get_individual(level=level, num_signal_events_per_set=num_signal_events_per_set)
                    .train_model(remake_datasets=remake_datasets)
                )       
        
    def evaluate_all(self, remake_datasets, epoch="final"):

        for num_signal_events_per_set in Numbers_of_Signal_Events_per_Set().tuple_:
            for level in Names_of_Levels().tuple_:
                self.results[level][num_signal_events_per_set] = (
                    self.get_individual(
                        level=level, 
                        num_signal_events_per_set=num_signal_events_per_set
                    ).evaluate_model(
                        remake_datasets=remake_datasets, 
                        epoch=epoch
                    )
                )

    def evaluate_subset(self, levels, nums_signal_events_per_set, remake_datasets, epoch="final"):
        
        for num_signal_events_per_set in nums_signal_events_per_set:
            for level in levels:
                self.results[level][num_signal_events_per_set] = (
                    self.get_individual(
                        level=level, 
                        num_signal_events_per_set=num_signal_events_per_set
                    ).evaluate_model(
                        remake_datasets=remake_datasets, 
                        epoch=epoch
                    )
                )

    def plot_image_examples_all(self, remake_datasets):
        for level in Names_of_Levels().tuple_:
            for num_events_per_set in Numbers_of_Signal_Events_per_Set().tuple_:
                (
                    self.get_individual(level=level, num_signal_events_per_set=num_events_per_set)
                    .plot_image_examples(remake_datasets=remake_datasets)
                )

    def plot_image_examples_subset(self, levels, nums_events_per_set, remake_datasets):
        for level in levels:
            for num_events_per_set in nums_events_per_set:
                (
                    self.get_individual(level=level, num_signal_events_per_set=num_events_per_set)
                    .plot_image_examples(remake_datasets=remake_datasets)
                )         

    def get_individual(self, level, num_signal_events_per_set):
        return self.group[level][num_signal_events_per_set]
        
       


class Event_by_Event_Group:

    def __init__(
        self,
        num_evaluation_sets_per_label,
        num_evaluation_sets_per_label_sensitivity,
        q_squared_veto,
        std_scale,
        shuffle,
        uniform_label_counts,
        loss_fn,
        learning_rate,
        learning_rate_scheduler_reduction_factor,
        size_of_training_batch,
        size_of_evaluation_batch,
        number_of_epochs,
        number_of_epochs_between_checkpoints,
        results_table,
        device
    ):
        
        self.initialize_group(
            num_evaluation_sets_per_label=num_evaluation_sets_per_label,
            num_evaluation_sets_per_label_sensitivity=num_evaluation_sets_per_label_sensitivity,
            q_squared_veto=q_squared_veto,
            std_scale=std_scale, 
            shuffle=shuffle,
            uniform_label_counts=uniform_label_counts,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            learning_rate_scheduler_reduction_factor=learning_rate_scheduler_reduction_factor,
            size_of_training_batch=size_of_training_batch,
            size_of_evaluation_batch=size_of_evaluation_batch,
            number_of_epochs=number_of_epochs,
            number_of_epochs_between_checkpoints=number_of_epochs_between_checkpoints,
            results_table=results_table,
            device=device
        )

        self.num_evaluation_sets_per_label = num_evaluation_sets_per_label

        self.possible_levels = (Names_of_Levels().generator, Names_of_Levels().detector)

    def train_all(self, remake_datasets):
        for level in (Names_of_Levels().generator, Names_of_Levels().detector):
            self.get_individual(level).train_model(remake_datasets)

    def train_subset(self, levels, remake_datasets):
        for level in levels:
            (
                self.get_individual(level)
                .train_model(remake_datasets=remake_datasets)
            )       

    def evaluate_all(self, remake_datasets, epoch="final"): 

        self.results = {
            level : {
                num_events_per_set : self.get_individual(level).evaluate_model(
                        num_events_per_set=num_events_per_set, 
                        remake_datasets=remake_datasets, 
                        epoch=epoch
                    )
                for num_events_per_set in Numbers_of_Signal_Events_per_Set().tuple_
            }
            for level in self.possible_levels
        }

    def evaluate_subset(self, levels, nums_events_per_set, remake_datasets, epoch="final"):

        for level in levels:
            for num_events_per_set in nums_events_per_set:
                (
                    self.get_individual(level)
                    .evaluate_model(
                        num_events_per_set=num_events_per_set, 
                        remake_datasets=remake_datasets,
                        epoch=epoch
                    )
                )

    def plot_results_all(self):
        
        _, axs = plt.subplots(
            len(self.possible_levels), 
            len(Numbers_of_Signal_Events_per_Set().tuple_), 
            sharex=True, 
            sharey=True
        )

        for (level, num_events_per_set), ax in zip(product(self.possible_levels, Numbers_of_Signal_Events_per_Set().tuple_), axs.flat):
            
            plot_linearity(
                linearity_test_results=self.results[level][num_events_per_set].linearity_results, 
                model_settings=self.get_individual(level).model_settings,
                dataset_settings=self.get_individual(level).evaluation_set_datasets_settings[num_events_per_set],
                path_to_plots_dir=Paths_to_Directories().path_to_plots_dir,
                ax=ax,
                save=False
            )

        plt.savefig(Paths_to_Directories().path_to_plots_dir.joinpath("ebe_linearity_grid.png"), bbox_inches="tight")
        plt.close()

    def get_individual(self, level):
        return self.group[level]

    def initialize_group(
        self,
        num_evaluation_sets_per_label,
        num_evaluation_sets_per_label_sensitivity,
        q_squared_veto,
        std_scale,
        shuffle,
        uniform_label_counts,
        loss_fn,
        learning_rate,
        learning_rate_scheduler_reduction_factor,
        size_of_training_batch,
        size_of_evaluation_batch,
        number_of_epochs,
        number_of_epochs_between_checkpoints,
        results_table,
        device
    ):
        
        num_evaluation_sets_per_label = to_dict_over_num_signal_events_per_set(num_evaluation_sets_per_label)
        
        common_parameters = dict(
            num_evaluation_sets_per_label=num_evaluation_sets_per_label,
            num_evaluation_sets_per_label_sensitivity=num_evaluation_sets_per_label_sensitivity,
            q_squared_veto=q_squared_veto,
            std_scale=std_scale,
            shuffle=shuffle,
            uniform_label_counts=uniform_label_counts,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            learning_rate_scheduler_reduction_factor=learning_rate_scheduler_reduction_factor,
            size_of_training_batch=size_of_training_batch,
            size_of_evaluation_batch=size_of_evaluation_batch,
            number_of_epochs=number_of_epochs,
            number_of_epochs_between_checkpoints=number_of_epochs_between_checkpoints,
            results_table=results_table,
            device=device
        )

        self.group = {
            level : Event_by_Event(
                level=level,
                **common_parameters
            )
            for level in (Names_of_Levels().generator, Names_of_Levels().detector)
        }