
from ..constants import Names_of_Splits, Raw_Signal_Trial_Ranges


def get_raw_signal_trial_range(split):
        
    if split not in Names_of_Splits().tuple_:
        raise ValueError
    
    range_ = (
        Raw_Signal_Trial_Ranges().train if (split == Names_of_Splits().train)
        else Raw_Signal_Trial_Ranges().validation if (split == Names_of_Splits().validation)
        else None
    )
    if range_ is None: raise ValueError
    return range_


def calc_num_signal_bkg_per_set(num_signal_events_per_set, bkg_fraction):

    num_bkg_events_per_set = (
        int((bkg_fraction / (1-bkg_fraction)) * num_signal_events_per_set) 
        if bkg_fraction is not None 
        else 0
    )
    return num_signal_events_per_set, num_bkg_events_per_set


# def calc_num_signal_bkg_per_set(num_events_per_set, bkg_fraction):

#     num_events_per_set_bkg = (
#         int(num_events_per_set * bkg_fraction) 
#         if bkg_fraction is not None
#         else 0
#     )
#     num_events_per_set_signal = (
#         num_events_per_set 
#         - num_events_per_set_bkg
#     )
#     return num_events_per_set_signal, num_events_per_set_bkg