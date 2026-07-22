import pandas as pd
from distribution import getDistribution, getCategories
from imblearn.over_sampling import SMOTE
import warnings

"""
This file contains functions to sample the data after it has been split between test, validation, 
and training sets and before the model is trained. This can be done using oversampling, hybrid
sampling, or SMOTE. SMOTE information used can be found at 
https://imbalanced-learn.org/stable/index.html
## gcong Keeps the second most populous category constant and adjusts the amount of other categories. 
For continuous data, we MIGHT want to try to keep the most populous category constant
"""


def categorize(data, categoryColumnName = "rating"):
    """
    Find the categories present in data

    Parameters
    ----------
    data : pd.DataFrame
        Data
    categoryColumnName : str
        Name of column in data with category information

    Returns
    -------
    categories : float list
        List of the values present in categoryColumnName
    categoriesNumber : int
        Number of categories
    initialValues : pd.Series
        Pandas series where categories are the index and the values are the number of times each 
        category occurs in data
    """
    initialValues = data[categoryColumnName].value_counts()

    categories = list(initialValues.index)
    categoriesNumber = len(categories)
    
    entries = len(data)

    sortedCategories = categories.copy()
    sortedCategories.sort()

    distribution = pd.DataFrame(index = sortedCategories)
    distribution["Occurrences"] = initialValues
    distribution["Distribution"] = distribution["Occurrences"] / entries

    print("Initial Distribution\n")
    print(distribution)
    print("Entries:", entries)
    print()

    return categories, categoriesNumber, initialValues


def getRatios(ratios, categoriesNumber):
    """
    Get the ratios to be used for sampling

    Parameters
    ----------
    ratios : float list
        List of ratios that sum to 1 or a list with 99 as a single entry indicating that a uniform
        distribution should be used
    categoriesNumber : int
        Number of categories

    Returns
    -------
    ratios : float list
        List of ratios that sum to 1
    """
    if ratios[0] == 99 and len(ratios) == 1:
        ratios = [(1 / categoriesNumber) for i in range(categoriesNumber)]

    if len(ratios) != categoriesNumber:
        warnings.warn("Number of categories does not equal number of ratios")

    return ratios


def getSmoteRatios(ratios, categories, categoriesNumber, initialValues):
    ratios = getRatios(ratios, categoriesNumber)
    
    anchorValue = initialValues.iloc[0] / ratios[0]
    smoteRatios = dict()

    for (i, category) in enumerate(categories):
        smoteRatios[category] = int(anchorValue * ratios[i])

    return smoteRatios


def sample(oldData, ratios, categories, initialValues, anchor, anchorIndex, categoryColumnName):
    """
    Sample the data as specified

    Parameters
    ----------
    oldData : pd.DataFrame
        Unsampled Data
    ratios : float list
        Ratio of each category to be used when sampling
    categories : float list
        Categories
    initialValues : pd.Series
        Count of how many occurrences in oldData of each category
    anchor : float
        Category to remain unchanged
    anchorIndex : int
        Index of anchor within categories
    categoryColumnName : str
        Name of column in data with category information

    Returns
    -------
    data : pd.DataFrame
        Sampled Data
    """
    allCategories = categories.copy()
    allCategories.sort()

    categories.remove(anchor)
    data = oldData.loc[oldData[categoryColumnName] == anchor]
    anchorRatio = ratios[anchorIndex]
    ratios.pop(anchorIndex)
    anchorGoal = len(data.loc[data[categoryColumnName] == anchor])
    totalData = int(anchorGoal / anchorRatio)

    for category in categories:

        currentIndex = categories.index(category)
        goal = int(totalData * ratios[currentIndex])

        factor = int(goal / initialValues[category])
        remainder = goal - (factor * initialValues[category])
        rows = oldData.loc[oldData[categoryColumnName] == category]

        concatList = [rows for i in range(factor)]
        if remainder != 0:
            concatList.append(rows.sample(n = remainder))
        concatList.insert(0, data)
        data = pd.concat(concatList)

    print("Resampled Distribution\n")
    finalValues = getDistribution(data, allCategories, categoryColumnName)["Occurrences"]

    return data


def oversampling(data, ratios, categoryColumnName):
    """
    Oversample data according to ratios

    Parameters
    ----------
    data : pd.DataFrame
        Unsampled Data
    ratios : float list
        List of ratios that sum to 1 or a list with 99 as a single entry indicating that a uniform
        distribution should be used
    categoryColumnName : str
        Name of column in data with category information

    Returns
    -------
    sample(...) : pd.DataFrame
        Sampled Data
    """
    print("Oversamppling is being used.\n")

    categories, categoriesNumber, initialValues = categorize(data, categoryColumnName)

    anchor = categories[0]
    anchorIndex = categories.index(anchor)

    ratios = getRatios(ratios, categoriesNumber)

    return sample(data, ratios, categories, initialValues, anchor, anchorIndex, categoryColumnName)


def hybrid(data, ratios, categoryColumnName):
    """
    Hybrid sample data according to ratios

    Parameters
    ----------
    data : pd.DataFrame
        Unsampled Data
    ratios : float list
        List of ratios that sum to 1 or a list with 99 as a single entry indicating that a uniform
        distribution should be used
    categoryColumnName : str
        Name of column in data with category information

    Returns
    -------
    sample(...) : pd.DataFrame
        Sampled Data
    """
    print("Hybrid sampling is being used\n")

    categories, categoriesNumber, initialValues = categorize(data, categoryColumnName)

    anchor = categories[1]
    anchorIndex = categories.index(anchor)

    ratios = getRatios(ratios, categoriesNumber)

    return sample(data, ratios, categories, initialValues, anchor, anchorIndex, categoryColumnName)


def smote(data, ratios, categoryColumnName):
    """
    Use SMOTE to oversample according to ratios

    Parameters
    ----------
    data : pd.DataFrame
        Unsampled Data
    ratios : float list
        List of ratios that sum to 1 or a list with 99 as a single entry indicating that a uniform
        distribution should be used
    categoryColumnName : str
        Name of column in data with category information

    Returns
    -------
    sampledData : pd.DataFrame
        Sampled Data
    """
    print("SMOTE is being used\n")

    categories, categoriesNumber, initialValues = categorize(data, categoryColumnName)
    categoryColumn = data[categoryColumnName]

    ratios = getSmoteRatios(ratios, categories, categoriesNumber, initialValues)

    sampledData = SMOTE(sampling_strategy = ratios).fit_resample(data, categoryColumn)[0]

    print("Resampled Distribution\n")
    finalValues = getDistribution(sampledData, categories, categoryColumnName)["Occurrences"]

    return sampledData

