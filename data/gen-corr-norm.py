import numpy as np

cov_mat = [[1, 0.5, 0.5],
           [0.5, 1, 0.7],
           [0.5, 0.7, 1]]

from scipy.stats import norm
def get_correlated_normals_in_interval(size, intervals, std):
    corr_normal_values = get_correlated_normal_distributions(len(intervals), size, std)
    resized_normal_values = []
    for i in range(0, len(intervals)):
        resized_normal_values.append(resize_normal_zero_mean(corr_normal_values[i],
            intervals[i][0], intervals[i][1]))
    return resized_normal_values



def trunctated_normal(size, min, max, std):
    mean = (min + max) / 2
    r = np.zeros(size)
    count = 0
    while count < len(r):
        rn = norm.rvs(mean, std)
        if rn >= min and rn <= max:
            r[count] = round(rn, 2)
            count += 1
    return r


def get_correlated_normal_distributions(number, size, std):
    norm_values = np.zeros(shape=(number, size))
    for i in range(0, number):
        # norm_values[i] = trunctated_normal(size, 0, 1, 1)
        norm_values[i] = norm.rvs(0, 1 / std, size=size)
    return np.dot(get_matrix_for_correlating_values(cov_mat), norm_values)


from scipy.linalg import eigh, cholesky
def get_matrix_for_correlating_values(cov_mat):
    # Compute the eigenvalues and eigenvectors.
    evals, evecs = eigh(cov_mat)
    # Construct c, so c*c^T = r.
    return cholesky(cov_mat, lower=True)
    # return np.dot(evecs, np.diag(np.sqrt(evals)))


def resize_normal_zero_mean(norm_values, min, max):
    l = []
    for val in norm_values:
        val = (val + 1) / 2
        l.append(val * (max - min) + min)
    return np.array(l)

if __name__ == "__main__":
    norm_values_corr = get_correlated_normal_distributions(3, 500, 3)
    intervals = [[10, 20], [15, 20], [15, 20]]
    for i in range(0, 3):
        norm_values_corr[i] = resize_normal_zero_mean(norm_values_corr[i],
            intervals[i][0], intervals[i][1])

    print(norm_values_corr)

    corr_in_interv = get_correlated_normals_in_interval(500, [[10, 20], [15, 20], [15, 20]], 3)


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