import numpy as np


def mean_causal_effect(mu_1: np.array, mu_0: np.array) -> float:
    """
    Compute the absolute mean standardized causal effect between two potential outcomes.
    
    This aims at measuring how much the treatment changes the outcome on a population sample.

    .. math:: \Delta_{\mu} = \frac{1}{N} \sum_{i=1}^N | \frac{\mu_{1}(x_i) - \mu_{0}(x_i)}{\mu_{0}(x_i)} |

    Args:
        mu_1 (np.array): _description_
        mu_0 (np.array): _description_

    Returns:
        float: _description_
    """
    return np.abs(mu_1/mu_0 - 1).mean()


def mean_causal_effect_symmetric(mu_1: np.array, mu_0: np.array) -> float:
    """
    Compute the absolute mean standardized causal effect between two potential outcomes. This version is symmetric.
    
    This aims at measuring how much the treatment changes the outcome on a population sample.

    .. math:: \Delta-sym_{\mu} = \frac{1}{N} \sum_{i=1}^N  \frac{|\mu_{1}(x_i) - \mu_{0}(x_i)|}{|\mu_{0}(x_i) + \mu_{1}(x_i) - \frac{1}{N} \sum_{i=j}^N\mu_{0}(x_j) + \mu_{1}(x_j)|}

    Args:
        mu_1 (np.array): _description_
        mu_0 (np.array): _description_

    Returns:
        float: _description_
    """
    
    mean_sum = (mu_1 + mu_0).mean()

    return (np.abs(mu_1 - mu_0) / np.abs(mu_1+mu_0 - mean_sum)).mean()
    


def mean_causal_effect_symmetric_variante(mu_1: np.array, mu_0: np.array) -> float:
    """
    Compute the absolute mean standardized causal effect between two potential outcomes. This version is symetric and puts the individual participation of each sample only on the denominator.
    
    This aims at measuring how much the treatment changes the outcome on a population sample.

    .. math:: \Delta-sym2_{\mu} = \frac{\frac{1}{N} \sum_{i=1}^N [|\mu_{1}(x_i) - \mu_{0}(x_i)|]}{\frac{1}{N} \sum_{i=1}^N [|\mu_{0}(x_i) + \mu_{1}(x_i) - \frac{1}{N} \sum_{i=j}^N[\mu_{0}(x_j) + \mu_{1}(x_j)]|]}

    Args:
        mu_1 (np.array): _description_
        mu_0 (np.array): _description_

    Returns:
        float: _description_
    """
    abs_mean_diff = np.abs(mu_1 - mu_0).mean()
    mean_sum = (mu_1 + mu_0).mean()

    return abs_mean_diff / np.abs(mu_1+mu_0 - mean_sum).mean()



def mean_causal_variation(mu_1: np.array, mu_0: np.array) -> float:
    """
    Compute the normalized standard deviation of the causal effect. 
    
    This aims at measuring how much the treatment effect varies on a population sample, ie. the complexity of the effect. It is a bit like the heterogeneity score but on the whole population instead of by propensity score bins.

    .. math:: \mathcal[H]_{\mu} = \hat_{var}(\mu_1 - \mu_0) / \hat_{var}(\mu_1+\mu_0)

    Args:
        mu_1 (np.array): _description_
        mu_0 (np.array): _description_

    Returns:
        float: _description_
    """
    sd_causal_effect = np.var(mu_1 - mu_0)
    normalization = np.var(mu_1 + mu_0)
    return sd_causal_effect / normalization
    
