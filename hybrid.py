import drugMatrixCompletion as dmc
import numpy as np
import pandas as pd
import argparse
from stats import printBreak

"""
This file tries random ratio combinations to try and surver the space of possible hybrid sampling
approaches for drugMatrixCompletion.
"""


def getRandomRatios(length):
    """
    Return an array of random numbers that sum to 1

    Parameters
    ----------
    length : int
        The length of the array to return

    Returns
    -------
    randomRatios : np.array
        An array of length length composed of random entires that sum to 1
    """
    randomNumbers = np.random.random(length)
    randomRatios = randomNumbers / randomNumbers.sum()
    randomRatios = list(randomRatios)
    randomRatios.sort(reverse = True)

    return randomRatios


def runExperiment(data, min_rating, max_rating, masterTrain, masterVal, test, args):
    """
    Runs SVD with randomized ratios and compiles the results

    Parameters
    ----------
    data : pd.DataFrame
        All of the data loaded in the Netflix Challenge format
    min_rating : float
        Minimum value in data.rating
    max_rating : float
        Maximum value in data.rating
    masterTrain : pd.DataFrame
        Portion of data to be used for training
    mastrVal : pd.DataFrame
        Portion of data to be used for validation
    test : pd.DataFrame
        Portion of data to be held out for testing
    args : Namespace
        The arguments that were collected from the shell script can be referenced as features of
        the return Namespace
    """
    categories = list(data[args.categoryColumnName].unique())
    print("Number of Categories:", len(categories))

    columns = categories.copy()
    columns.append("Test MAE")
    results = pd.DataFrame(columns = columns)

    runs = int(args.ratios[0])
    for i in range(runs):
        printBreak()
        print("Run-" + str(i))

        ratios = getRandomRatios(len(categories))
        individualResults = ratios.copy()
        print("Ratios:",ratios,"\n")

        train, val = dmc.sampleData(masterTrain.copy(), masterVal.copy(), args.sampling, 
            args.valDrop, ratios, args.r1, args.categoryColumnName)

        svd = dmc.runSVD(train, val, args.sampling, args.lr, args.wd, args.epochs, args.factors, args.stop, 
            min_rating, max_rating)

        mae = dmc.predictTest(svd, test, args.categoryColumnName, args.excelFile, args.minmaxFile, 
            args.testStats)

        individualResults.append(mae)
        results.loc[i] = individualResults

    printBreak()

    print("Total Results\n")
    print(results)

def main():

    args = dmc.arguments()

    data = dmc.loadData(args.path, args.categoryColumnName)
    min_rating, max_rating = dmc.maxMinRating(data)

    dummyMatrix = pd.DataFrame
    masterTrain, masterVal, test = dmc.splitData(data, args.full, args.unbalanced, args.r1, args.r2, 
        args.testfile, dummyMatrix)

    runExperiment(data, min_rating, max_rating, masterTrain, masterVal, test, args)


if __name__ == "__main__":
    main()
