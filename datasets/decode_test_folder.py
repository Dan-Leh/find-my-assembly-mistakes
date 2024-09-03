import os


def decode_test_sets(test_set_names:str, roi_crop:bool, center_roi:bool) -> dict:
    ''' From the names listed in the 'test_sets' command line argument, get
    the path to the folder of each test dataset to run on
    
    Arguments:
        test_set_names (str): a string containing the name of each test set
        roi_crop (bool): whether to use region-of-interest crops.
        
    Returns:
        test_configuration (dict of strings) -- with following key:value pairs:
            'set_name': list of all set names
            'test_type': list of each test type (eg. testing for performance with
                varying orientation difference, or translation difference)
            'data_path': list of the path to each json file containing information
                about all image pairs in the test set
            'more_nqd_bins': if true, have plot the results using the 11 nQD ranges
                from 0 to 1 nQD, else use the previous 4 bins: 0, 0-0.1, 0.1-0.2 
                and 0.2-1
        '''

    eval_set_list = []
    if 'v1' in test_set_names:
        eval_set_list.append('Val_states')
    if 'v2' in test_set_names:
        eval_set_list.append('Val_states_v2')
    if 'pp' in test_set_names:
        eval_set_list.append('PseudoPlane')
    if 'extra' in test_set_names:
        eval_set_list.append('Val_states_extra')
    if 'inter' in test_set_names:
        eval_set_list.append('Val_states_inter')
    if 'new_parts' in test_set_names:
        eval_set_list.append('New_parts')       
        
    data_root = "/shared/nl011006/res_ds_ml_restricted/dlehman/SyntheticData"
    
    set2dir_name = {"Val_states": "Val_states_set_200x1000", 
                    "PseudoPlane": "PseudoPlane_200x1000",
                    "Val_states_v2": "Val_states_v2",
                    "Val_states_extra": "Val_states_extra",
                    "Val_states_inter": "Val_states_inter",
                    "New_parts": "Val_states_new_parts"}



    test_configuration = {'set_name': [], 'test_type': [], 'data_path': [],
                          'more_nqd_bins': []}
    # Extract all the test sets that we want to evaluate on
    for eval_set in eval_set_list:
        # Run evaluation script for every aspect/type that is tested for (eg. nQD)
        if roi_crop:  # whether to test on ROI crops or full images
            test_type_list = ['ROI']
            if eval_set == 'New_parts': 
                test_type_list += ['ROI_missing', 'ROI_present']
        else:
            test_type_list = ['Orientation', 'Scale', 'Translation', 
                            'Missing', 'Present']
        for test_type in test_type_list:      
            new_bins = False  # default
            if test_type == 'Orientation':
                data_list_filepath = os.path.join(data_root, set2dir_name[eval_set], 
                                                "eval_rand_20k.json")
            elif test_type == 'Translation' or test_type == 'Scale':
                data_list_filepath = os.path.join(data_root, "Test_pairs", 
                                        eval_set+"_"+test_type, "pair_info.json")
            elif test_type == 'Missing':
                data_list_filepath = os.path.join(data_root, set2dir_name[eval_set], 
                                                "eval_missing_parts_2k.json")
            elif test_type =='Present':
                data_list_filepath = os.path.join(data_root, set2dir_name[eval_set], 
                                                "eval_present_parts_2k.json")
            elif test_type.startswith('ROI'):
                if center_roi:  # if the roi crops should be aligned
                    if test_type == 'ROI_missing':
                        data_list_filepath = os.path.join(data_root, 
                                set2dir_name[eval_set], "eval_missing_parts_30k.json")
                    elif test_type == 'ROI_present':
                        data_list_filepath = os.path.join(data_root, 
                                set2dir_name[eval_set], "eval_present_parts_30k.json")
                    else:
                        data_list_filepath = os.path.join(data_root, 
                                set2dir_name[eval_set], "eval_rand_30k.json")
                    test_type += "_aligned"
                else: # if the crops can have some small translations
                    # check if there is a json file containing crops:
                    if test_type == 'ROI':
                        try_path = os.path.join(data_root, set2dir_name[eval_set], 
                                                "eval_rand_33k_w_crops.json")
                        if os.path.exists(try_path):
                            data_list_filepath = try_path
                            new_bins = True
                        else: 
                            print(f'Path not found {try_path}')
                            data_list_filepath = os.path.join(data_root, "Test_pairs", 
                                            eval_set+"_"+test_type, "pair_info.json")
                    else:
                        mispres = 'missing' if test_type.endswith('missing') else 'present'
                        try_path = os.path.join(data_root, set2dir_name[eval_set], 
                                                f"eval_{mispres}_parts_33k_w_crops.json")
                        if os.path.exists(try_path):
                            data_list_filepath = try_path
                            new_bins = True
                        else: 
                            print(f'Path not found {try_path}')
                            data_list_filepath = os.path.join(data_root, "Test_pairs", 
                                            eval_set+"_"+test_type, "pair_info.json")
                
            test_configuration['set_name'].append(eval_set)
            test_configuration['test_type'].append(test_type)
            test_configuration['data_path'].append(data_list_filepath)
            test_configuration['more_nqd_bins'].append(new_bins)
    
    return test_configuration
