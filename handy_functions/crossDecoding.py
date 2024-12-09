import numpy as np
from itertools import combinations

def get_trial_num(cond_binned):
    # Initialize a repeat matrix
    repeat_matrix = np.zeros((10, 3))  
    rm_pos = 0  

    # Process individual columns for unique value counts
    for col in [1, 2]:  
        for unique in np.unique(cond_binned[:, col]):  
            # Get trials corresponding to the unique value
            trls = np.unique(cond_binned[cond_binned[:, col] == unique][:, 0]).shape[0]
            
            # Record information in the matrix
            repeat_matrix[rm_pos] = [col, unique, trls]
            rm_pos += 1

    # Process column 3 for unique pair counts
    col = 3  
    for val1, val2 in combinations(np.unique(cond_binned[:, col]), 2):  
        # Get rows for each value in the pair
        rows_val1 = np.unique(cond_binned[cond_binned[:, col] == val1][:, 0])  
        rows_val2 = np.unique(cond_binned[cond_binned[:, col] == val2][:, 0])  

        # Calculate shared trials and store in the matrix
        shared_trials = rows_val1.shape[0] + rows_val2.shape[0]
        repeat_matrix[rm_pos] = [col, val1 + val2, shared_trials]
        rm_pos += 1

    # Get the minimum trial count
    num_trials_subsample = repeat_matrix[:, 2].min()

    return num_trials_subsample

def get_trials(cond_binned,num_trls,num_repeats=1,condition='contrast',cond_ind=0,repeat=0):
    np.random.seed(42)
    
    trial_matrix = []
    if condition == 'contrast':
        condition_names = ["lo", "hi", "mix"]
        unique_conditions = np.sort(np.unique(cond_binned[:, 1]))
        condition_trials = []

        # Collect trials for each condition
        for c in unique_conditions:
            trls = np.unique(cond_binned[cond_binned[:, 1] == c][:, 0])
            subsamples = np.array([np.random.choice(trls, size=int(num_trls), replace=False) for _ in range(int(num_repeats))]).T
            condition_trials.append(subsamples)
            trial_matrix.append(subsamples)

        # Generate 50/50 mix from each pair of conditions
        for i in range(len(unique_conditions) - 1):  # Pairwise combinations of conditions
            for j in range(i + 1, len(unique_conditions)):
                condition_a = condition_trials[i]
                condition_b = condition_trials[j]

                # Create 50/50 mix
                mixed_matrix = np.array([
                    np.random.choice(np.concatenate([condition_a[:, k], condition_b[:, k]]), size=int(num_trls), replace=False)
                    for k in range(int(num_repeats))
                ]).T

                trial_matrix.append(mixed_matrix)

    elif condition=='speed':
        condition_names = ["slow", "fast", "mix"]
        unique_conditions = np.sort(np.unique(cond_binned[:, 2]))
        condition_trials = []

        # Collect trials for each condition
        for c in unique_conditions:
            trls = np.unique(cond_binned[cond_binned[:, 2] == c][:, 0])
            subsamples = np.array([np.random.choice(trls, size=int(num_trls), replace=False) for _ in range(int(num_repeats))]).T
            condition_trials.append(subsamples)
            trial_matrix.append(subsamples)

        # Generate 50/50 mix from each pair of conditions
        for i in range(len(unique_conditions) - 1):  # Pairwise combinations of conditions
            for j in range(i + 1, len(unique_conditions)):
                condition_a = condition_trials[i]
                condition_b = condition_trials[j]

                # Create 50/50 mix
                mixed_matrix = np.array([
                    np.random.choice(np.concatenate([condition_a[:, k], condition_b[:, k]]), size=int(num_trls), replace=False)
                    for k in range(int(num_repeats))
                ]).T

                trial_matrix.append(mixed_matrix)
    
    elif condition == 'AV':
        condition_names = ["lo", "hi", "mix"]
        rank_grp = np.where(cond_binned[:,6] > 1193, 2, 1)
        unique_conditions = np.sort(np.unique(rank_grp)) 
        condition_trials = []

        # Collect trials for each condition
        for c in unique_conditions:
            trls = np.unique(cond_binned[rank_grp == c][:, 0])
            subsamples = np.array([np.random.choice(trls, size=int(num_trls), replace=False) for _ in range(int(num_repeats))]).T
            condition_trials.append(subsamples)
            trial_matrix.append(subsamples)

        # Generate 50/50 mix from each pair of conditions
        for i in range(len(unique_conditions) - 1):  # Pairwise combinations of conditions
            for j in range(i + 1, len(unique_conditions)):
                condition_a = condition_trials[i]
                condition_b = condition_trials[j]

                # Create 50/50 mix
                mixed_matrix = np.array([
                    np.random.choice(np.concatenate([condition_a[:, k], condition_b[:, k]]), size=int(num_trls), replace=False)
                    for k in range(int(num_repeats))
                ]).T

                trial_matrix.append(mixed_matrix)

    else:
        # direction
        condition_names = []
        for d1, d2 in combinations(np.sort(np.unique(cond_binned[:, 3])), 2):  
            condition_name = f"{int(d1)}_{int(d2)}"
            condition_names.append(condition_name)  # Add the condition name to the list

            rows_d1 = np.unique(cond_binned[cond_binned[:, 3] == d1][:, 0])  
            rows_d2 = np.unique(cond_binned[cond_binned[:, 3] == d2][:, 0]) 

            subsamples_d1 = np.array([np.random.choice(rows_d1, size=int(num_trls/2), replace=False) for _ in range(int(num_repeats))]).T
            subsamples_d2 = np.array([np.random.choice(rows_d2, size=int((num_trls/2)+1), replace=False) for _ in range(int(num_repeats))]).T

            subsamples = np.concatenate((subsamples_d1, subsamples_d2), axis=0)
            trial_matrix.append(subsamples)

    trls = np.sort(trial_matrix[cond_ind][:,repeat])
    row_log = np.isin(cond_binned[:, 0], trls)

    return trls, row_log, condition_names[cond_ind]
