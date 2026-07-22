import pandas as pd
import numpy as np
import datetime

"""
This file replicates the functions fetch_ml_ratings and _ml_ratings_csv_to_df from funk_SVD.dataset
with changes to allow for an additional column for categories
"""


def fetch_ml_ratings(filename, categoryColumnName = "rating"):
    """
    Replicates the functions fetch_ml_ratings and _ml_ratings_csv_to_df from funk_SVD.dataset

    Parameters
    ----------
    filename : str
        Path to where the data is located; data must be saved as a CSV and have the columns u_id,
        i_id, rating, timestamp, and the column listed as categoryColumnName (if it is not rating)
    categoryColumnName : str
        Column in the data that contains the category information

    Returns
    -------
    df : pd.DataFrame
        Data in the format required by funk_SVD library
    """
    names = ['u_id', 'i_id', 'rating', 'timestamp']
    dtype = {'u_id': np.uint32, 'i_id': np.uint32, 'rating': np.float64}

    if categoryColumnName != "rating":
        names.append(categoryColumnName)
        dtype[categoryColumnName] = np.float64

    df = pd.read_csv(filename, names = names, dtype = dtype, header = 0, sep = ',')

    columns = ['u_id', 'i_id', 'rating']
    if categoryColumnName != "rating":
        columns.append(categoryColumnName)

    df = df[columns]

    df.reset_index(drop=True, inplace=True)

    return df
