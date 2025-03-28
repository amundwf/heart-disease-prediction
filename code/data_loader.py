import pandas as pd
import os

# Data loading functions for the N500 dataset

def load_subject_features(features_dir, splits_path, subset="all_dbs"):
    # subset: str. Values: "db1", "db2", "all_dbs".
    # Example input values:
    # features_dir:
    #  features_dir = os.path.join("..", "data", "calculated_features", "N500_all_features")
    # splits_path:
    #  data_dir = os.path.join("..", "data", "dataset_splits")
    #  splits_folder = "generated_splits_2"
    #  splits_path = os.path.join(data_dir, splits_folder)

    # Determine filenames based on the subset
    if subset == "db1":
        filenames = ("db1_training_set.csv", "db1_validation_set.csv", "db1_test_set.csv")
    elif subset == "db2":
        filenames = ("db2_training_set.csv", "db2_validation_set.csv", "db2_test_set.csv")
    elif subset == "all_dbs":
        filenames = ("all_dbs_training_set.csv", "all_dbs_validation_set.csv", "all_dbs_test_set.csv")
    else:
        raise ValueError("The 'subset' variable must be set to 'db1', 'db2' or 'all_dbs'.")

    # Build paths for the splits
    paths = [os.path.join(splits_path, fname) for fname in filenames]
    df_splits_subjects = [pd.read_csv(path) for path in paths] # [training set subjects, validation set subjects, test set subjects]

    data_splits = [] # List to hold the DataFrames for each split
    for subjects_df in df_splits_subjects:
        list_of_feature_dfs = []

        for _, subj_row in subjects_df.iterrows(): # for each subject in the split
            db_name = subj_row['database']
            if db_name == 'Fantasia':
                db_name = 'FD'
            
            subject_id = subj_row['subject_id']
            subject_features_file = f'{db_name}_{subject_id}_features.csv'
            path_subject_features = os.path.join(features_dir, db_name, subject_features_file)

            if os.path.exists(path_subject_features):
                try:
                    df_features = pd.read_csv(path_subject_features)
                    list_of_feature_dfs.append(df_features) # Add the subject's features to the list
                except FileNotFoundError:
                    print(f"Error: Feature CSV not found for subject {subject_id} in {db_name} at {path_subject_features}")
                except pd.errors.EmptyDataError:
                    print(f"Warning: Feature CSV is empty for subject {subject_id} in {db_name} at {path_subject_features}")
            else:
                print(f"Warning: Feature CSV path does not exist: {path_subject_features}")

        if list_of_feature_dfs:
            features_df = pd.concat(list_of_feature_dfs, ignore_index=True) # Concatenate the subjects' features into one Dataframe
        else:
            print(f"Warning: No features found for split.")
        
        data_splits.append(features_df)

    df_train = data_splits[0]
    df_val = data_splits[1]
    df_test = data_splits[2]
    return df_train, df_val, df_test
