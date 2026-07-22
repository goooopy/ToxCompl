from fetchMLRatings import fetch_ml_ratings
from funk_svd import SVD
import pandas as pd
from sendToJSON import save, load
import argparse
from distribution import getDistribution, getCategories
from sampling import oversampling, hybrid, smote
from stats import getPrediction, filterPrediction, getStatistics, printBreak
from saveToExcel import saveToExcel
from sklearn.metrics import mean_absolute_error
import viewLatentFactors as vlf
from sklearn.preprocessing import MinMaxScaler

"""
Created by Guojing Congg

Updated by Charlie Murphy
24 June 2022

This file performs matrix completion on data in the Netflix challenge format using Funk SVD.
Statistics on the result are then computed on the test data.
"""


def arguments():
    """
    Get the necessary arguments from a shell script via argparse

    Returns
    -------
    args : Namespace
        The arguments that were collected from the shell script can be referenced as features of
        the return Namespace
    """
    parser = argparse.ArgumentParser(description='drugMatrixCompletion.py')
    parser.add_argument("--lr", type=float, default=0.01,
            help="learning rate")
    parser.add_argument("--epochs", type=int, default=300,
            help="number of training epochs")
    parser.add_argument("--factors", type=int, default=300,
            help="factors")
    parser.add_argument("--wd", type=float, default=0.01,
            help="Weight decay")
    parser.add_argument("--r1", type=int, default=7,
            help="seed")
    parser.add_argument("--r2", type=int, default=8,
            help="seed")
    parser.add_argument("--path", type=str)
    parser.add_argument("--predfile", type=str, default='predictions.json',
            help="predictfile to write")
    parser.add_argument("--full", type=int, default=0,
            help="whether use the full train set")
    parser.add_argument("--sampling", type=str, default="normal",
            help="normal/oversampling/undersampling/hybrid")
    parser.add_argument("--testfile", type=str, default="testData.csv")
    parser.add_argument("--ratios", type=float, nargs="+", default=[99], 
        help="99 represents a default")
    parser.add_argument("--stop", type=int, default=1,
        help="early stopping")
    parser.add_argument("--unbalanced", type=str, default="balanced", 
        help="use if the input file is already sampled;'balanced' represents default behavior")
    parser.add_argument("--valDrop", type=int, default=0,
        help="0 = no sampling, 1 = sampled same as train, 2 = sampled with train")
    parser.add_argument("--excelFile", type=str, default="excelOutput.txt")
    parser.add_argument("--saveData", type=int, default=0)
    parser.add_argument("--testStats", type=int, default=1)
    parser.add_argument("--categoryColumnName", type=str, default="rating")
    parser.add_argument("--latentFactors", type=str, default="Latent-Factors-Saved.npz")
    parser.add_argument("--minmaxFile", type = str, default = "drugmatrix-minmax-scaler-details.json")

    args = parser.parse_args()
    print(args)

    return args


def loadData(filename, categoryColumnName = "rating"):
    """
    Load the data from a CSV file in the Netflix Challenge format to the necessary data structures
    for Funk-SVD

    Parameters
    ----------
    filename : str
        Path to a CSV file with columns u_id : int, i_id : int, and rating : float; a final 
        category with name categoryColumnName and type float may be present for category 
        information, otherwise rating will be used
    categoryColumnName : str
        Name of column in data with category information

    Returns
    -------
    data : pd.DataFrame
        A pandas DataFrame with the same columns as above
    """
    data = fetch_ml_ratings(filename, categoryColumnName)

    return data


def maxMinRating(df):
    """
    Find the minimum and maximum rating found in the DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with an int-vakued column rating

    Returns
    -------
    min_rating : int
        The minimum value in the rating column of df
    max_rating : int
        The maximum value in the rating column of df
    """
    min_rating = df["rating"].min()
    max_rating = df["rating"].max()

    print('min_rating', min_rating, 'max_rating', max_rating)

    return min_rating, max_rating


def createEmptyMatrix(unbalanced, df):
    """
    Creates an DataFrame with rating values of 0 that has the same dimensions as the input 
    DataFrame

    Parameters
    ----------
    unbalanced : str
        This string is either 'balanced' in which case the DataFrame df is used to find the 
        dimensions or the path to a file that has a DataFrame whose dimensions should be used
        instead
    df : pd.DataFrame
        The DataFrame whose dimensions are used if unbalanced = 'balanced'

    Returns
    -------
    df_full : pd.DataFrame
        A DataFrame with columns u_id : int, i_id : int, and rating : float that
        has max_uid * max_idd entries where every (u_id, i_id) pair is unique and every entry for
        rating is 0.
    originalMatrix : pd.DataFrame
        If unbalanced != 'balanced', then this is the matrix found in this file. Otherwise, it is
        an empty DataFrame
    max_uid : int
        The largest u_id present in df_full
    max_iid : int
        The largest i_id present in df_full
    """
    if unbalanced == "balanced":
        originalMatrix = pd.DataFrame
        pd_uid = df["u_id"]
        pd_iid = df["i_id"]

    else:
        originalMatrix = loadData(unbalanced)
        pd_uid = originalMatrix["u_id"]
        pd_iid = originalMatrix["i_id"]

    max_uid = pd_uid.max() +1 
    max_iid = pd_iid.max() +1 

    print('max_uid', max_uid, 'max_iid', max_iid)
    print()

    uid_l = []
    iid_l = []
    for i in range(max_uid):
        tmp_l = [i]*max_iid
        uid_l += tmp_l 
        tmp_l = list(range(max_iid))
        iid_l += tmp_l

    dict_full = {'u_id':uid_l, 'i_id':iid_l, 'rating':[0.0]*len(iid_l)}
    df_full=pd.DataFrame(dict_full)

    print('df_full', df_full)

    return df_full, originalMatrix, max_uid, max_iid


def splitData(df, full, unbalanced, r1, r2, testFileName, originalMatrix, saveData = 0):
    """
    Splits a DataFrame into training, validation, and testing sets

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame beign split into training, validation, and testing
    full : int
        If full = 1, then all of df is used for training. Otherwise, only 90% of df is used to make
        the training data with the remainder split evenly between validation and testing
    unbalaced : str
        If unbalanced != 'balanced' then the originalMatrix is used for the testing data
    r1 : int
        The random seed used to sample df for the training data
    r2 : int
        The random seed used to sample the validation data
    testFileName : str
        The path where a CSV file of the testing data will be saved
    originalMatrix : pd.DataFrame
        The DataFrame that the testing data will be sampled from if unbalanced != 'balanced'
    saveData : int
        Whether testing data is saved to a CSV file

    Returns
    -------
    train : pd.DataFrame
        Training Data
    val : pd.DataFrame
        Validation Data
    test : pd.DataFrame
        Testing Data
    """
    if full == 0:
        print('not using the full set to train')
        #df = df.sample(frac=1).reset_index(drop=True) # shuffle the whole thing
        train = df.sample(frac=0.9, random_state=r1)
        val = df.drop(train.index.tolist()).sample(frac=0.5, random_state=r2)
        test = df.drop(train.index.tolist()).drop(val.index.tolist())

    else:
        print('using the full set to train')
        train = df
        val = df.sample(frac=0.1, random_state = r1)
        test = df.sample(frac=0.1, random_state = r2)

    if unbalanced != "balanced":
        train2 = originalMatrix.sample(frac=0.9, random_state=r1)
        val2 = originalMatrix.drop(train2.index.tolist()).sample(frac=0.5, random_state=r2)
        test = originalMatrix.drop(train2.index.tolist()).drop(val2.index.tolist())

    print(train)
    print()

    if saveData:
        test.to_csv(testFileName)

        print("Test Data Saved to", testFileName)
        print()

    return train, val, test


def sampleData(train, val, sampling, valDrop, ratios, r1, categoryColumnName):
    """
    Oversample of hybrid sample the data if necessary

    Parameters
    ----------
    train : pd.DataFrame
        Training Data
    val : pd.DataFrame
        Validation Data
    sampling : str
        How to sample the data; should be one of 'hybrid', 'normal', 'oversampling', 
        'undersampling', or 'SMOTE'
    valDrop : int
        Whether or not sample validation data; should be one of 0, 1, or 2
    ratios : int list
        The list of ratios of the desired distribution of the categories after sampling
    r1 : int
        The random seed used to sample df for the training data
    categoryColumnName : str
        Name of column in data with category information

    Returns
    -------
    train : pd.DataFrame
        Training Data
    val : pd.DataFrame
        Validation Data
    """
    # Sample only training data
    if valDrop == 0:

        if sampling == "oversampling":
            train = oversampling(train, ratios, categoryColumnName)
        elif sampling == "hybrid":
            train = hybrid(train, ratios, categoryColumnName)
        elif sampling == "SMOTE":
            train = smote(train, ratios, categoryColumnName)

    # Sample training and validation data separately
    elif valDrop == 1:

        if sampling == "oversampling":
            train = oversampling(train, ratios, categoryColumnName)
            val = oversampling(val, ratios, categoryColumnName)
        elif sampling == "hybrid":
            train = hybrid(train, ratios, categoryColumnName)
            val = hybrid(val, ratios, categoryColumnName)
        elif sampling == "SMOTE":
            train = smote(train, ratios, categoryColumnName)
            val = smote(val, ratios, categoryColumnName)

    # Sample training and validation data together
    # Note that this method will likely cause overlap in the training and validation sets
    elif valDrop == 2:

        combinedData = pd.concat([train, val])

        if sampling == "oversampling":
            combinedData = oversampling(combinedData, ratios, categoryColumnName)
        elif sampling == "hybrid":
            combinedData = hybrid(combinedData, ratios, categoryColumnName)
        elif sampling == "SMOTE":
            combinedData = smote(combinedData, ratios, categoryColumnName)

        if sampling in ["oversampling", "hybrid", "SMOTE"]:
            trainFrac = 18 / 19
            train = combinedData.sample(frac = trainFrac, random_state = r1)
            val = combinedData.drop(list(set(train.index.tolist())))
        else:
            train = combinedData

    return train, val


def runSVD(train, val, sampling, lr, wd, epochs, factors, stop, min_rating, max_rating):
    """
    Runs the Funk-SVD algorithm

    Parameters
    ----------
    train : pd.DataFrame
        Training Data
    val : pd.DataFrame
        Validation Data
    sampling : str
        How to sample the data; should be one of 'hybrid', 'normal', 'oversampling',
        'undersampling', or 'SMOTE
    lr : float
        Learning Rate
    wd : float
        Weight Decay
    epochs : int
        Epochs
    factors : int
        Factors
    stop : int
        0 corresponds to not stop early and 1 corresponds to stopping early
    min_rating : int
        The minimum value in the rating column of df
    max_rating : int
        The maximum value in the rating column of df

    Returns
    -------
    svd : SVD
        Object that has been trained on data and can be used to predict the full matrix
    """
    if sampling == "undersampling":
        print("Undersampling is being used.\n")
        undersampling = True
    else:
        undersampling = False

    columns = ["u_id", "i_id","rating"]
    train = train[columns]
    val = val[columns]

    print('running with different hyper-parameters:')
    print('===factor: ', factors, "wd:", wd, "====")

    svd = SVD(lr=lr, reg=wd, n_epochs=epochs, n_factors=factors, early_stopping=stop,
            shuffle=False, min_rating=min_rating, max_rating=max_rating, min_delta=0.0001, 
            undersampling = undersampling)

    svd.fit(X=train, X_val=val)

    return svd


def unscale(data, userIds, minimums, maximums):
    """
    Reverse minmax scaling

    Parameters
    ----------
    data : float list
        List of data to unscale
    minimum : float
        Minimum unscaled rating
    maximum : float
        Maxium unscaled rating

    Returns
    -------
    result : float list
        List of data that is no longer scaled
    """
    result = []
    for i in range(len(data)):
        minimum = minimums[userIds[i]]
        maximum = maximums[userIds[i]]
        result.append(data[i] * (maximum - minimum) + minimum)

    return result


def getMAE(test, pred, minimums, maximums):
    """
    Find the unscaled test MAE

    Parameters
    ----------
    test : pd.DataFrame
        Ground truth; must have column rating
    pred : float lsit
        Predictions
    minimums : float
        Minimum unscaled rating
    maximums : float
        Maxium unscaled rating

    Returns
    -------
    mae : float
        Mean Absolute Error
    """
    userIds = list(test.u_id)
    pred = unscale(pred, userIds, minimums, maximums)
    testUnscaled = unscale(list(test.rating), userIds, minimums, maximums)

    mae = mean_absolute_error(testUnscaled, pred)
    return mae


def predictTest(svd, test, categoryColumnName, excelFilename, minmaxFilename, allStats):
    """
    Calculates statistics on the accuracy of the Funk-SVD algorithm on testing data that was not
    used in training

    Parameters
    ----------
    svd : SVD
        Object that has been trained on data and can be used to predict the full matrix
    test : pd.DataFrame
        Testing Data
    categoryColumnName : str
        Name of column in data with category information,
    excelFilename : str
        Path to save txt file
    minmaxFilename : str
        Path to JSON file with min-max scaler information
    allStats : bool
        Whether to print all statistics or just MAE
    """
    pred = svd.predict(test)

    print("Test Data", test[:50])
    print()

    print("Test Variance:", test.rating.var())
    print("Test Mean:", test.rating.mean())
    print()

    print("Test Prediction:", pred[:50])

    printBreak()

    categories = getCategories(test, categoryColumnName)

    if allStats:

        test_dict={'max_uid': test.shape[0], 'max_iid': 1, 'pred':pred}
        predictedData = getPrediction(test_dict)

        predictedData = predictedData.round(0)

        print("Test Data\n")
        test = test.reset_index()
        distribution = getDistribution(test, categories)

        print("Predicted Data (Test) \n")
        predictedDistribution = getDistribution(predictedData, categories)

        print("Test Statistics\n")
        stats, accuracy, mae, rmse = getStatistics(test.rating, distribution["Occurrences"], 
            predictedData.rating, predictedDistribution["Occurrences"], categories)

        saveToExcel(excelFilename, predictedDistribution, stats, accuracy, mae, rmse)

    else:

        scalerInformation = load(minmaxFilename)
        minimums = scalerInformation["min"]
        maximums = scalerInformation["max"]

        mae = getMAE(test, pred, minimums, maximums)
        print("Test MAE:",mae,"\n")

        if categoryColumnName != "rating":
            print("Test MAE by Category\n")

            maeByCategory = pd.Series(index = categories, dtype = float)

            for category in categories:
                subset = test.loc[test[categoryColumnName] == category]
                subsetPrediction = svd.predict(subset)

                maeByCategory[category] = getMAE(subset, subsetPrediction, minimums, maximums)

            print(maeByCategory)

    return mae


def predictFullMatrix(svd, df_full, max_uid, max_iid, filename, saveData = 0):
    """
    Predict the entire matrix and save the results to a JSON file.

    Parameters
    ----------
    svd : SVD
        Object that has been trained on data and can be used to predict the full matrix
    df_full : pd.DataFrame
        Empty version of the full matrix
    max_uid : int
        The largest u_id present in df_full
    max_iid : int
        The largest i_id present in df_full
    filename : str
        Path to where the JSON file should be saved
    saveData : int
        Whether to save predictions to a JSON file

    Returns
    -------
    pred : int list
        List of predictions to the entire matrix
    """
    pred = svd.predict(df_full)

    printBreak()

    print('full pred', pred[:50])
    print()

    print("Shape:")
    print(len(pred))
    print()

    if saveData:
        final_dict={'max_uid': max_uid, 'max_iid':max_iid, 'pred':pred}
        save(filename, final_dict)

    return pred


def main():

    args = arguments()

    df = loadData(args.path, args.categoryColumnName)
    min_rating, max_rating = maxMinRating(df)
    df_full, originalMatrix, max_uid, max_iid = createEmptyMatrix(args.unbalanced, df)

    train, val, test = splitData(df, args.full, args.unbalanced, args.r1, args.r2, args.testfile,
        originalMatrix, args.saveData)
    train, val = sampleData(train, val, args.sampling, args.valDrop, args.ratios, args.r1, 
        args.categoryColumnName)

    svd = runSVD(train, val, args.sampling, args.lr, args.wd, args.epochs, args.factors, args.stop, 
        min_rating, max_rating)
    
    predictTest(svd, test, args.categoryColumnName, args.excelFile, args.minmaxFile, 
        args.testStats)

    vlf.view()

    if args.saveData:
        predictFullMatrix(svd, df_full, max_uid, max_iid, args.predfile, args.saveData)

        vlf.saveAsNPZ(args.latentFactors)


if __name__ == "__main__":
    main()