import numpy as np
from distribution import getDistribution, getCategories
from stats import getPrediction, filterPrediction, getStatistics, printBreak
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_gaussian_process, plot_convergence
import drugMatrixCompletion as dmc
import warnings
import argparse

"""
Created by Charlie Murphy
14 July 2022

This file uses Bayesian optimization to find the optimal distribution for hybrid sampling.
Additional information about the Bayesian optimization package used available at 
https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html and 
https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html
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
    parser = argparse.ArgumentParser(description='bayesianOptimization')
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
    parser.add_argument("--stop", type=int, default=1,
        help="early stopping")
    parser.add_argument("--seed", type = int, default = 1234)
    parser.add_argument("--n", type = int, default = 10)
    parser.add_argument("--initial", type = int, default = None)
    parser.add_argument("--method", type = str, default = "normalized")
    parser.add_argument("--acquisitionFunction", type = str, default = "EI")
    parser.add_argument("--goal", type = str, default = "averageF1")

    args = parser.parse_args()
    print(args)

    return args


def prepareData(args):
    """
    Load in data using drugMatrixCompletion.py that will be used for every run

    Parameters
    -------
    args : Namespace
        The arguments that were collected from the shell script can be referenced as features of
        the return Namespace

    Returns
    -------
    train : pd.DataFrame
        Training data
    val : pd.DataFrame
        Validation data
    test : pd.DataFrame
        Testing data
    min_rating : float
        Minimum value of the rating column of the DataFrame loaded from args.path
    max_rating : float
        Maximum value of the rating column of the DataFrame loaded from args.path
    """
    df = dmc.loadData(args.path, "rating")
    min_rating, max_rating = dmc.maxMinRating(df)
    df_full, originalMatrix, max_uid, max_iid = dmc.createEmptyMatrix("balanced", df)

    train, val, test = dmc.splitData(df, args.full, "balanced", args.r1, args.r2, "dummy",
        originalMatrix, 0)

    return train, val, test, min_rating, max_rating


def getMajorityRatio(data):
    """
    Find the fraction of the training data that is the majority class

    Parameters
    ----------
    data : pd.DataFrame
        Data; has column rating

    Returns
    -------
    majorityRatio : float
        Fraction of the training data that is the majority class
    categories : float list
        Categories present in data.rating
    """
    categories = getCategories(data)
    distribution = getDistribution(data, categories)
    majorityRatio = distribution["Distribution"].max()

    return majorityRatio, categories


def calculateTestStats(svd, test):
    """
    Find the average test F1 of the test data using the model svd

    Parameters
    ----------
    svd : funk_svd.SVD
        Matrix completion model
    test : 
        Testing data

    Returns
    -------
    averageF1 : float
        Average test F1 score
    minorityAverageF1 : float
        Average test F1 score, excluding the F1 score of the majority class
    minimumF1 : float
        Minimum test F1 score
    precision : float
        Average test precision, excluding the preicison of the majority class
    recall :float
        Average test recall, excluding the recall of the majority class
    """
    pred = svd.predict(test)

    categories = getCategories(test, "rating")
    test_dict={'max_uid': test.shape[0], 'max_iid': 1, 'pred':pred}
    predictedData = getPrediction(test_dict)

    predictedData = predictedData.round(0)

    test = test.reset_index()
    distribution = getDistribution(test, categories)

    predictedDistribution = getDistribution(predictedData, categories)

    stats, accuracy, mae, rmse = getStatistics(test.rating, distribution["Occurrences"], 
        predictedData.rating, predictedDistribution["Occurrences"], categories)

    # Remove Row with majority class
    minorityStats = stats.drop(index = stats.index[2])

    averageF1 = stats.F1.mean()
    minorityAverageF1 = minorityStats.F1.mean()
    minimumF1 = stats.F1.min()
    precision = minorityStats.Precision.mean()
    recall = minorityStats.Recall.mean()

    print("\nAverage F1:",averageF1)
    print("Minority Average F1:",minorityAverageF1)
    print("Minimum F1:",minimumF1)

    print("\nMinority Average Precision:",precision)
    print("Minority Average Recall:",recall)

    return averageF1, minorityAverageF1, minimumF1, precision, recall


def runSVD(ratios, masterTrain, masterVal, test, min_rating, max_rating, majorityRatio, args):
    """
    Use SVD to create a model given the ratios of hybrid sampling

    Parameters
    ----------
    ratios : float list
        Ratios to use in hybrid sampling
    masterTrain : pd.DataFrame
        Training data before sampling
    masterVal : pd.DataFrame
        Validation data before sampling
    test : pd.DataFrame
        Testing data
    min_rating : float
        Minimum value of the rating column of the DataFrame loaded from args.path
    max_rating : float
        Maximum value of the rating column of the DataFrame loaded from args.path
    majorityRatio : float
        Fraction of the training data that is the majority class
    args : Namespace
        The arguments that were collected from the shell script can be referenced as features of
        the return Namespace

    Returns
    -------
    goal : float
        The result of the criteria that is being optimized as per args.goal. 0 may be returned if 
        the distribution given is at risk of oversampling the majority class.
    """
    # Prevent oversampling of the majority class
    if ratios[0] > majorityRatio or ratios[1] < 0.05 or 0.0 in ratios:
        return 0

    printBreak()
    print("Ratios:", ratios,"\n")

    train, val = dmc.sampleData(masterTrain.copy(), masterVal.copy(), "hybrid", 2, ratios, 
        args.r1, "rating")

    svd = dmc.runSVD(train, val, "hybrid", args.lr, args.wd, args.epochs, args.factors, args.stop, 
        min_rating, max_rating)

    averageF1, minorityAverageF1, minimumF1, precision, recall = calculateTestStats(svd, test)

    if args.goal == "averageF1":
        return averageF1
    elif args.goal == "minorityF1":
        return minorityAverageF1
    elif args.goal == "minimumF1":
        return minimumF1
    elif args.goal == "precision":
        return precision
    elif args.goal == "recall":
        return recall
    else:
        warnings.warn("Invalid Goal. Valid goals are 'averageF1', 'minorityF1', 'minimumF1', 'precision', and 'recall'.")
    

def getRatios(x):
    """
    Convert list chosen by optimizer into one that sums to 1

    Parameters
    ----------
    x : float list
        List of floats between 0 and 1
    
    Returns
    -------
    ratios : float list
        List of floats that sum to 1 and have the same ratio to one another as x
    """
    x = np.array(x.copy())
    ratios = list(x / sum(x))

    return ratios


def printResults(results):
    """
    Prints results from Bayesian optimization

    Parameters
    ----------
    results : scipy object
        Resulting object from running gp_minimize
    """
    printBreak()
    print(results)
    printBreak()

    maxAverageF1 = 1 - results.fun
    print("Maximum Goal Value:",maxAverageF1,"\n")
    print("Optimal Distribution:",getRatios(results.x))


def normalized(args, masterTrain, masterVal, test, min_rating, max_rating, majorityRatio, categories):
    """
    The inputs are floats between 0 and 1 that are normalized to get the distributions.

    Parameters
    ----------
    args : Namespace
        The arguments that were collected from the shell script can be referenced as features of
        the return Namespace
    masterTrain : pd.DataFrame
        Training data before sampling
    masterVal : pd.DataFrame
        Validation data before sampling
    test : pd.DataFrame
        Testing data
    min_rating : float
        Minimum value of the rating column of the DataFrame loaded from args.path
    max_rating : float
        Maximum value of the rating column of the DataFrame loaded from args.path
    majorityRatio : float
        Fraction of the training data that is the majority class

    Returns
    -------
    f : function
        Function to be optimized using Bayesian Optimization
    dimensions : int tuple list
        List of bounds for each parameter of f
    intitalPoint : int list
        The first set of parameters to try. This list must be the same length as dimensions and
        each point must be within the bounds of its corresponding tuple in dimensions
    """
    # Bayesian optimization minimizes f, so averageF1 is subtracted from its maximum of 1
    def f(x):
        ratios = getRatios(x)
        averageF1 = runSVD(ratios, masterTrain, masterVal, test, min_rating, max_rating, majorityRatio, 
            args)
        return 1 - averageF1

    dimensions = [(0.0, 1.0) for i in range(len(categories))]

    initialPoint = [0.7, 0.13, 0.13, 0.02, 0.02]

    return f, dimensions, initialPoint


def nonNormalized(args, masterTrain, masterVal, test, min_rating, max_rating, majorityRatio):
    """
    The inputs are the ratios for the minority classes. The inputs are not normalized, but their
    sum must be less than 1.

    Parameters
    ----------
    args : Namespace
        The arguments that were collected from the shell script can be referenced as features of
        the return Namespace
    masterTrain : pd.DataFrame
        Training data before sampling
    masterVal : pd.DataFrame
        Validation data before sampling
    test : pd.DataFrame
        Testing data
    min_rating : float
        Minimum value of the rating column of the DataFrame loaded from args.path
    max_rating : float
        Maximum value of the rating column of the DataFrame loaded from args.path
    majorityRatio : float
        Fraction of the training data that is the majority class

    Returns
    -------
    f : function
        Function to be optimized using Bayesian Optimization
    dimensions : int tuple list
        List of bounds for each parameter of f
    intitalPoint : int list
        The first set of parameters to try. This list must be the same length as dimensions and
        each point must be within the bounds of its corresponding tuple in dimensions
    """
    # Bayesian optimization minimizes f, so averageF1 is subtracted from its maximum of 1
    def f(x):
        ratios = x.copy()
        ratios.insert(0, 1 - sum(x))
        averageF1 = runSVD(ratios, masterTrain, masterVal, test, min_rating, max_rating, majorityRatio, 
            args)
        return 1 - averageF1

    dimensions = [(0.0, 0.3), (0.0, 0.3), (0.0, 0.1), (0.0, 0.1)]

    initialPoint = [0.13, 0.13, 0.02, 0.02]

    return f, dimensions, initialPoint


def samples(args, masterTrain, masterVal, test, min_rating, max_rating, majorityRatio):
    """
    The inputs are the number of data points to be sampled as opposed to a distribution.

    Parameters
    ----------
    args : Namespace
        The arguments that were collected from the shell script can be referenced as features of
        the return Namespace
    masterTrain : pd.DataFrame
        Training data before sampling
    masterVal : pd.DataFrame
        Validation data before sampling
    test : pd.DataFrame
        Testing data
    min_rating : float
        Minimum value of the rating column of the DataFrame loaded from args.path
    max_rating : float
        Maximum value of the rating column of the DataFrame loaded from args.path
    majorityRatio : float
        Fraction of the training data that is the majority class

    Returns
    -------
    f : function
        Function to be optimized using Bayesian Optimization
    dimensions : int tuple list
        List of bounds for each parameter of f
    intitalPoint : int list
        The first set of parameters to try. This list must be the same length as dimensions and
        each point must be within the bounds of its corresponding tuple in dimensions
    """
    # Bayesian optimization minimizes f, so averageF1 is subtracted from its maximum of 1
    def f(x):
        ratios = getRatios(x)
        averageF1 = runSVD(ratios, masterTrain, masterVal, test, min_rating, max_rating, majorityRatio, 
            args)
        return 1 - averageF1

    dimensions = [(1000000, 62719534), (1000000, 4000000), (1000000, 4000000), (100000, 4000000), (100000, 4000000)]

    initialPoint = [15048632, 2794746, 2794746, 429960, 429960]

    return f, dimensions, initialPoint


def main():

    args = arguments()

    masterTrain, masterVal, test, min_rating, max_rating = prepareData(args)

    majorityRatio, categories = getMajorityRatio(masterTrain)

    if args.method == "normalized":
        f, dimensions, initialPoint = normalized(args, masterTrain, masterVal, test, min_rating, 
            max_rating, majorityRatio, categories)

    elif args.method == "non-normalized":
        f, dimensions, initialPoint = nonNormalized(args, masterTrain, masterVal, test, min_rating, 
            max_rating, majorityRatio)

    elif args.method == "samples":
        f, dimensions, initialPoint = samples(args, masterTrain, masterVal, test, min_rating, 
            max_rating, majorityRatio)

    else:
        warnings.warn("Invalid Method. Valid methods are 'normalized', 'non-normalized', and 'samples'.")

    results = gp_minimize(f, dimensions, acq_func = args.acquisitionFunction, n_calls = args.n, 
        n_initial_points = args.initial, random_state = args.seed, x0 = initialPoint)

    printResults(results)


if __name__ == "__main__":
    main()