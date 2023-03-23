import os
import time
import numpy as np
import pandas as pd
from scipy.special import comb
import itertools
import multiprocessing as mp


def calc_enum_data(i, all_compositions, law, n):
    ind = all_compositions[i]
    law_list = [np.tile(law.iloc[j].to_numpy(), (max(ind[j], 0), 1)) for j in range(n)]
    new_law = np.vstack(law_list)

    epsilon = 1e-12  # Add a small constant value to avoid division by zero
    cov_matrix = np.cov(new_law[:, 0], new_law[:, 1])
    stddevs = np.sqrt(np.diag(cov_matrix)) + epsilon
    corrcoef = cov_matrix / np.outer(stddevs, stddevs)

    return (corrcoef[0, 1], np.prod(np.power(1 / n, ind)))


def generate_all_compositions(n):
    for c in itertools.combinations(range(2 * n - 1), n - 1):
        counts = np.diff((0,) + c + (2 * n - 1,))
        yield np.array([counts[i] - 1 if i < len(counts) - 1 else counts[i] for i in range(len(counts))])


def main():
    law = pd.read_csv('/Users/rockyauer/Downloads/COMP_STATS/law.csv')
    law_pre = law.drop([0, 10])
    n = len(law_pre)

    all_compositions = list(generate_all_compositions(n))

    enum_data_path = "enumData.npy"
    enum_data_csv_path = "enumData.csv"

    start_time = time.time()
    with mp.Pool(processes=4) as pool:
        enum_data = pool.starmap(calc_enum_data,
                                 [(i, all_compositions, law_pre, n) for i in range(len(all_compositions))])
    elapsed_time = time.time() - start_time

    enum_data = np.array(enum_data)
    np.save(enum_data_path, enum_data)

    # Convert the NumPy array to a DataFrame and save it as a CSV file
    enum_data_df = pd.DataFrame(enum_data, columns=["cor", "weight"])
    enum_data_df.to_csv(enum_data_csv_path, index=False)

    print("Elapsed time:", elapsed_time)


if __name__ == "__main__":
    main()