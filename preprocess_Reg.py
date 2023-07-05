import os
import pandas as pd
import numpy as np

RUNNING = 1.0
WALKING = 0.0

folder_path = 'fixed_data/'  # Update with the path to your folder

def load_dataset():
    """
    :return: this function return a dict of tuples:
    dict[i] = (df_of_steps, activity_type, steps_count)
    """
    max_len = 0
    file_list = os.listdir(folder_path)
    targets = dict()
    dataframes = dict()
    for file_index, file_name in enumerate(file_list):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    if i == 2:
                        if 'Run' in line.split(',')[1]:
                            activity = RUNNING
                        else:
                            activity = WALKING
                    elif i == 3:
                        steps = int(line.split(',')[1].strip().strip('"'))
                        break

            df = pd.read_csv(file_path, skiprows=range(5)).dropna()
            df['Norm'] = df.apply(lambda row: (float(row['ACC X'])**2 + float(row['ACC Y'])**2 + float(row['ACC Z'])**2)**0.5, axis=1)
            vec = create_aggregations(df, activity)
            dataframes[file_index] = vec
            if len(df) > max_len:
                max_len = len(df)
            targets[file_index] = steps
    return dataframes, targets


def create_aggregations(df, activity):

    # features for each original feature (x,y,z,norm): mean, std, min, max
    # additional features: total_time, steps_count
    num_features = 18  # 4 for each original feature + 2 new feature
    cols = ['ACC X', 'ACC Y', 'ACC Z', 'Norm']
    feat_vector = np.zeros(num_features)
    i = 0
    for c in cols:
        feat_vector[i] = df[c].mean()
        feat_vector[i+1] = df[c].std()
        feat_vector[i + 2] = df[c].min()
        feat_vector[i + 3] = df[c].max()
        i += 4
    feat_vector[16] = df[df.columns[0]].max()
    feat_vector[17] = activity

    return feat_vector
