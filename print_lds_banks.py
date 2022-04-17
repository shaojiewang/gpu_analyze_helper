import numpy as np
import pandas as pd

def compute_lds_bank(k0, m0, m1, k1, padding_per_m1):
    #assert padding_per_m1 % 2 == 0
    shape = (m0 * m1, k0 * k1)
    lds_pos = np.zeros(shape, dtype=int)
    for i_k0 in range(k0):
        for i_m0 in range(m0):
            for i_m1 in range(m1):
                for i_k1 in range(k1):
                    pos = i_k0 * m0 * (m1 + padding_per_m1) * k1 + \
                          i_m0 * (m1 * k1 + padding_per_m1) + \
                          i_m1 * k1 + \
                          i_k1
                    lds_pos[i_m0 * m1 + i_m1][i_k0 * k1 + i_k1] = pos

    lds_bank = (lds_pos // 2) % 32
    return lds_pos, lds_bank

def gen_pd_dict(lds_bank, k):
    lds_bank_dict = {}
    for i_k in range(k):
        lds_bank_dict[i_k] = lds_bank[:, i_k]

    lds_bank_df = pd.DataFrame(lds_bank_dict)
    lds_bank_df.to_excel("lds_bank.xlsx")
    return lds_bank_df

if __name__ == "__main__":
    k0 = 4
    k1 = 8
    m0 = 16
    m1 = 8
    m1_padding = 8
    k = k0 * k1
    lds_pos, lds_bank = compute_lds_bank(k0, m0, m1, k1, m1_padding)
    np.savetxt("lds_pos.txt", lds_pos, "%4d")
    np.savetxt("lds_bank.txt", lds_bank, "%4d")
    np.savetxt("lds_bank.csv", lds_bank, "%d", ";")

    lds_bank_df = gen_pd_dict(lds_bank, k)
    #print(lds_bank_df)
