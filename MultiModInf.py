import pandas as pd
import numpy as np
import itertools as iter
import statsmodels.api as sm


def cov_calc(var_mat, cor_mat):
    """
    Description:
    ------------
    A function to calculate a covariance matrix from a correlation
    matrix.

    Parameters:
    -----------
    var_mat: np.array
        a diagonal matrix with variable variances as diagonal elements.
    cor_mat: np.array
        a matrix of correlations among variables.

    Returns:
    --------
    a covariance matrix.
    """
    return np.dot(np.sqrt(var_mat), np.dot(cor_mat, np.sqrt(var_mat)))


def generate_cor_matrix(coefs, nb_predictors, colin_range=(0, 0.4)):
    """
    Description:
    ------------
    A function to generate a corelation matrix with multicolineatity.

    Parameters:
    -----------
    coefs: List
        a vector of correlations
        between the first variable (the response) and all
        other variables.
    nb_predictors:
        the number of considered predictors.
    colin_range: tuple
        You must specify the range of
        corelation between predictors to choose from.
    """

    cor_mat = np.zeros([nb_predictors + 1, nb_predictors + 1])
    cor_mat[0, 1:] = coefs
    for i, j in zip(range(1, nb_predictors), range(2, nb_predictors + 1)):
        cor_mat[i, j:] = np.random.uniform(
            colin_range[0], colin_range[1], cor_mat[i, j:].shape[0]
        )
    cor_mat = cor_mat + cor_mat.T
    np.fill_diagonal(cor_mat, 1)
    return cor_mat


def generate_multivariate_data(
    means, vars, coefs, sample_size=100, multicolinearity=False, colin_range=None
):
    """
    Description:
    ------------
    A function to generate multivariate normaly distributed data.

    Parameters:
    -----------
    means: List or np.array
        a vector of means to generate distributions.
    vars: List or np.array
        a vector of variances to generate distributions.
    coefs: List or np.array
        if multicolinearity = False, a vector of correlations
        between the first variable (the response) and all
        other variables.
        if multicolinearity = True, a matrix of dimension
        len(means)*len(means) of correlations among variables.
    sample_size: int
        the number of generated observations.
    multicolinearity: Boolean
        decides whether multicolinearity is considered in data.
    colin_range: None or tuple
        if multicolinearity = False, then colin_range must be None
        if multicolinearity = True, you must specify the range of
        corelation between predictors to choose from.

    Returns:
    --------
    a pandas.DataFrame containing a response variable y and len(means)-1
    predictor variables.
    """
    if multicolinearity:
        cor_mat = generate_cor_matrix(
            coefs=coefs, nb_predictors=len(coefs), colin_range=colin_range
        )
    else:
        cor_mat = np.identity(len(vars))
        cor_mat[0, 1:] = coefs
        cor_mat = cor_mat + cor_mat.T
        np.fill_diagonal(cor_mat, 1)
    var_mat = np.diag(vars)
    covs = cov_calc(var_mat, cor_mat)
    cols = ["y"] + ["x" + str(v) for v in range(1, len(vars))]
    df = pd.DataFrame(
        np.random.multivariate_normal(means, covs, sample_size), columns=cols
    )
    return df


def multi_model_inf(df, response, variables, combinations="all"):
    """
    Description:
    ------------
    A function to calculate AIC metrics across all possible
    models.

    Parameters:
    -----------
    df: pandas.DataFrame
        the data
    response: str
        the name of the response variable in the data.
    variables: List of str
        the names of the predictor variables in the data.

    Returns:
    --------
    a pandas.DataFrame containing all possible models, their estimated parameters
    and their AIC metrics.
    """
    if combinations == "all":
        # find all predictor combinations
        comb = (iter.combinations(variables, l) for l in range(len(variables) + 1))
    else:
        # only consider models with a maximum number of predictors of 'combinations'
        comb = (iter.combinations(variables, l) for l in range(combinations + 1))
    model_variables = list(iter.chain.from_iterable(comb))
    # fit all possible models
    cols = ["model", "intercept"]
    cols.extend(variables)
    cols.extend(["AIC", "deltaAIC"])
    models = pd.DataFrame(columns=cols, dtype="float")
    models["model"] = [list(p) for p in model_variables]
    for i, predictors in enumerate(models["model"]):
        y = df[response]
        X = df[predictors]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        models.loc[i, "AIC"] = model.aic
        models.loc[i, "intercept"] = model.params["const"]
        predictors = model.params.drop("const")
        models.loc[i, predictors.index] = predictors.values
    # get deltaAIC values
    models = models.sort_values(by="AIC").reset_index(drop=True)
    models.loc[0, "deltaAIC"] = 0
    for i in range(1, models.shape[0]):
        models.loc[i, "deltaAIC"] = models.loc[i, "AIC"] - models.loc[0, "AIC"]
    # get AIC weights
    exp_aic = np.exp(-0.5 * models["deltaAIC"])
    models["AIC weight"] = exp_aic / np.sum(exp_aic)
    return models


def model_averaging(models, variables, method="full"):
    """
    Description:
    ------------
    A function to average model coefficients.

    Parameters:
    -----------
    models: pandas.dataFrame
        a dataframe containing model coeeficients and AIC weights.
    variables: List of str
        the names of the predictor variables in the data.
    method: str
        if "full", then full-averaging is performed.
        if "natural", then natural averaging is performed

    Returns:
    --------
    a dictionary of model-averaged estimated parameters.
    """
    avg = []
    param = ["intercept"] + variables
    for v in param:
        estimates = models[v].fillna(0)
        avg_estimate = (estimates * models["AIC weight"]).sum()
        avg.append(avg_estimate)
    return {v: a for v, a in zip(param, avg)}


def prediction(model_params, new_data):
    """
    Description:
    ------------
    A function to predict the response from new data
    using model (averaged) estimates.

    Parameters:
    -----------
    model_params: dict or pd.Series
        model (averaged) estimates.
    new_data: pd.DataFrame
        the test dataset

    Returns:
    --------
    a numpy array of predicted values
    """
    variables = list(model_params.keys())[1:]
    y_pred = np.sum(
        np.array([(model_params[v] * new_data[v]).tolist() for v in variables]), axis=0
    ) + np.repeat(model_params["intercept"], new_data.shape[0])
    return y_pred
