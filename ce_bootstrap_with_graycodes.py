import numpy as np
import pandas as pd
import itertools
from multiprocessing import Pool
from scipy.stats import multinomial

# define a function to generate Gray codes
def gray_codes(n):
    total = int(np.math.comb(2*n - 1, n - 1))
    gray_codes = np.zeros((total, n), dtype=int)
    r = np.zeros(n, dtype=int)
    r[0] = n
    t = n
    h = 0
    gray_codes[0, :] = r
    i = 1
    while r[n-1] != n:
        if t != 1:
            h = 0
        h = h + 1
        t = r[h-1]
        r[h-1] = 0
        r[0] = t - 1
        r[h] = r[h] + 1
        gray_codes[i, :] = r
        i = i + 1
    return gray_codes

def generate_all_compositions(n):
    gc = gray_codes(n)
    for code in gc:
        yield code

def calc_enum_data(i, n, law, all_compositions):
    ind = all_compositions[i]
    law_list = [np.tile(law.iloc[j].to_numpy(), (ind[j], 1)) for j in range(n)]
    new_law = np.vstack(law_list)
    corr = np.corrcoef(new_law[:, 0], new_law[:, 1])[0, 1]
    weight = multinomial.pmf(ind, n, [1 / n] * n)
    return (corr, weight)

def main():
    law_pre = pd.read_csv('/Users/rockyauer/Downloads/COMP_STATS/law.csv')
    law = law_pre.drop([0, 10])
    n = len(law)

    all_compositions = np.array(list(generate_all_compositions(n)))
    num_compositions = len(all_compositions)

    with Pool(4) as pool:
        enum_data = pool.starmap(calc_enum_data, [(i, n, law, all_compositions) for i in range(num_compositions)])

    enum_data_df = pd.DataFrame(enum_data, columns=['cor', 'weight'])
    enum_data_df.to_csv('enumData_gc.csv', index=False)

if __name__ == "__main__":
    main()