import numpy as np

"""
Assumed covariance matrix for cap-diameter, stem-height and stem-width to get a more realistic
simulation of mushrooms (mushrooms with larger caps -> mushrooms with higer stems)
The values are picked arbitrary and may be changed
"""
cov_mat = [[1, 0.5, 0.5],
           [0.5, 1, 0.7],
           [0.5, 0.7, 1]]

from scipy.stats import norm
def get_correlated_normals_in_interval(size, intervals, std):
    """
    Parameters
    ----------
    size: int
    number of random generated normal values per distribution
    intervals: list of lists of floats
    an min max interval for each generated normal distribution
    std: float
    standart deviation of the normal distributions

    Return
    ------------
    list of lists of floats
    each element is a list of size values representing a normal distribution in one interval

    Example
    ------------
    size = 353, intervals = [[10.0, 20.0], [15.0, 20.0], [15.0, 20.0]], std = 3
    -> return [[353 random normal values between 10.0 and 20.0], [353 random normal values between 15.0 and 20.0],
        [353 random normal values between 15.0 and 20.0]]
    """

    corr_normal_values = get_correlated_normal_distributions(len(intervals), size, std)
    resized_normal_values = []
    for i in range(0, len(intervals)):
        resized_normal_values.append(resize_normal_zero_mean(corr_normal_values[i],
            intervals[i][0], intervals[i][1]))
    return resized_normal_values


def get_correlated_normal_distributions(number, size, std):
    """
    Helper function of get_correlated_normals_in_interval()

    Parameters
    ----------
    number: int
    number of random generated normal distributions
    size: int
    number of random generated normal values per distribution
    std: float
    standart deviation of the normal distributions

    Return
    ------------
    list of lists of floats
    each element is a list size values representing a zero mean normal distribution with std,
    correlated to each other using the global covariance matrix cov_mat
    """

    norm_values = np.zeros(shape=(number, size))
    for i in range(0, number):
        norm_values[i] = norm.rvs(0, 1 / std, size=size)
    return np.dot(get_matrix_for_correlating_values("cholesky"), norm_values)


from scipy.linalg import eigh, cholesky
def get_matrix_for_correlating_values(method):
    """
    Helper function of get_correlated_normal_distributions()

    Parameters
    ------------
    method: str
    either "eigenvalues" or "cholesky" determining the used method

    Return
    ------------
    numpy.ndarray
    returns a matrix c from the matrix decomposition c*c^T = cov_mat

    Ressource: https://scipy-cookbook.readthedocs.io/items/CorrelatedRandomSamples.html
    """

    # Compute the eigenvalues and eigenvectors.
    evals, evecs = eigh(cov_mat)
    if method == "cholesky":
        return cholesky(cov_mat, lower=True)
    if method == "eigenvalues":
        return np.dot(evecs, np.diag(np.sqrt(evals)))


def resize_normal_zero_mean(norm_values, min, max):
    """
    Helper function of get_correlated_normals_in_interval()

    Parameters
    ----------
    norm_values: list of floats
    represents a zero mean normal distribution
    min: int
    lower interval border
    max: int
    upper interval border

    Return
    ------------
    list of floats
    the zero mean normal distribution resized to the fall symmetrically into the interval borders
    """

    l = []
    for val in norm_values:
        val = (val + 1) / 2
        l.append(val * (max - min) + min)
    return np.array(l)


if __name__ == "__main__":
    """
    Running this module results in an example run of creating normal sampled values for the metrical attributes
    cap-diameter, stem-height and stem-width using the values of the mushroom species 'Fly Agaric'.
    The three resulting normal distributions are then visualized with two plots:
    1) scatter plots showing the correlations between the attributes
    2) bar plot showing that the distribution is normal
    """

    size = 353
    norm_values_corr = get_correlated_normal_distributions(3, size, 3)
    intervals = [[10, 20], [15, 20], [15, 20]]
    for i in range(0, 3):
        norm_values_corr[i] = resize_normal_zero_mean(norm_values_corr[i],
            intervals[i][0], intervals[i][1])

    print(norm_values_corr)

    corr_in_interv = get_correlated_normals_in_interval(size, [[10, 20], [15, 20], [15, 20]], 3)


    # plot correlated and uncorrelated random samples
    from pylab import plot, show, axis, subplot, xlabel, ylabel, grid, hist
    import matplotlib.pyplot as plt
    subplot(1, 3, 1)
    plot(corr_in_interv[0], corr_in_interv[1], 'b.', c='grey')
    xlabel('cap diameter')
    ylabel('stem height')
    axis('equal')
    grid(True)

    subplot(1, 3, 2)
    plot(norm_values_corr[0], norm_values_corr[2], 'b.', c='grey')
    xlabel('cap diameter')
    ylabel('stem width')
    axis('equal')
    grid(True)

    subplot(1, 3, 3)
    plot(norm_values_corr[1], norm_values_corr[2], 'b.', c='grey')
    xlabel('stem height')
    ylabel('stem width')
    axis('equal')
    plt.tight_layout()
    grid(True)

    # improve spacing
    fig, ax = plt.subplots()
    plt.tight_layout()

    show()

    for i in range(0, 3):
        subplot(1, 3, i + 1)
        hist(norm_values_corr[i], color='grey')
        if i == 0:
            xlabel('cap diameter [10, 20]')
        if i == 1:
            xlabel('stem height [15, 20]')
        if i == 2:
            xlabel('stem width [15, 20]')
        grid(True)

    fig, ax = plt.subplots()
    plt.tight_layout()

    show()
