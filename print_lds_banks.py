import numpy as np
import pandas as pd
from sqlalchemy import column, true

class lds_positions:
    def __init__(self, k0, k1, m0, m1, padding_per_m1xk1, t_write_len, t_write_vec, t_read_vec) -> None:
        self.k0 = k0
        self.k1 = k1
        self.m0 = m0
        self.m1 = m1
        self.padding_per_m1xk1 = padding_per_m1xk1

        # ds write:
        self.t_write_len = t_write_len
        self.t_write_vec = t_write_vec

        # ds read:
        self.t_read_vec = t_read_vec

        self.lds_pos = None
        self.lds_bank = None
        self.ds_write_pos = None
        self.ds_write_bank = None
        self.ds_read_pos = None
        self.ds_read_bank = None

        self.block_size = 256

    def compute_lds_bank(self):
        #assert padding_per_m1 % 2 == 0
        m0 = self.m0
        m1 = self.m1
        k0 = self.k0
        k1 = self.k1
        padding_per_m1xk1 = self.padding_per_m1xk1
        shape = (m0 * m1, k0 * k1)
        lds_pos = np.zeros(shape, dtype=int)
        for i_k0 in range(k0):
            for i_m0 in range(m0):
                for i_m1 in range(m1):
                    for i_k1 in range(k1):
                        pos = i_k0 * m0 * (m1 * k1 + padding_per_m1xk1) + \
                              i_m0 * (m1 * k1 + padding_per_m1xk1) + \
                              i_m1 * k1 + \
                              i_k1
                        lds_pos[i_m0 * m1 + i_m1][i_k0 * k1 + i_k1] = pos

        lds_bank = (lds_pos // 2) % 32

        self.lds_bank = lds_bank
        self.lds_pos = lds_pos
        return lds_pos, lds_bank

    def gen_pd_dict(self):
        lds_bank = self.lds_bank
        k = self.k0 * self.k1
        lds_bank_dict = {}
        for i_k in range(k):
            lds_bank_dict[i_k] = lds_bank[:, i_k]

        lds_bank_df = pd.DataFrame(lds_bank_dict)
        lds_bank_df.to_excel("lds_bank.xlsx")
        return lds_bank_df

    def gen_ds_bank_dict(self, gen_write = true, gen_read = true):
        if gen_write:
            ds_write_bank = self.ds_write_bank
            ds_write_pos = self.ds_write_pos * 2

            col_t = []
            for i in range(self.block_size):
                col_t.append(f"t{i}")
            ds_write_bank_df = pd.DataFrame(ds_write_bank, columns=[0, 1], index=col_t)
            ds_write_pos_df = pd.DataFrame(ds_write_pos, columns=[0, 1], index=col_t)
            with pd.ExcelWriter('ds_write_bank.xlsx') as writer:  
                ds_write_bank_df.to_excel(writer, sheet_name='ds_write_bank')
                ds_write_pos_df.to_excel(writer, sheet_name='ds_write_pos')

        if gen_read:
            ds_read_bank = self.ds_read_bank
            ds_read_pos = self.ds_read_pos * 2

            lines = list(range(self.t_read_vec))

            col_t = []
            for i in range(self.block_size):
                col_t.append(f"t{i}")
            ds_read_bank_df = pd.DataFrame(ds_read_bank, columns=lines, index=col_t)
            ds_read_pos_df = pd.DataFrame(ds_read_pos, columns=lines, index=col_t)
            with pd.ExcelWriter('ds_read_bank.xlsx') as writer:  
                ds_read_bank_df.to_excel(writer, sheet_name='ds_read_bank')
                ds_read_pos_df.to_excel(writer, sheet_name='ds_read_pos')

    def gen_ds_write_pos_bank(self):
        t_write_len = self.t_write_len
        t_write_vec = self.t_write_vec
        block_size = self.block_size
        lds_pos = self.lds_pos
        write_shape = (block_size, t_write_vec)
        m = self.m0 * self.m1
        ds_write_pos = np.zeros(write_shape)
        for i_thread in range(block_size):
            m_pos = (i_thread * t_write_len) % m
            k_pos = (i_thread * t_write_len) // m * t_write_vec
            for i_vec in range(t_write_vec):
                k_pos = i_vec + k_pos
                ds_write_pos[i_thread][i_vec] = lds_pos[m_pos][k_pos]

        self.ds_write_pos = ds_write_pos
        self.ds_write_bank = (ds_write_pos // 2) % 32
        return ds_write_pos

    def gen_ds_read_pos(self):
        t_read_vec = self.t_read_vec
        block_size = self.block_size
        lds_pos = self.lds_pos
        read_shape = (block_size, t_read_vec)
        m = self.m0 * self.m1
        ds_read_pos = np.zeros(read_shape)
        for i_thread in range(block_size):
            i_n_pos = i_thread % 32
            i_n_block = i_thread % 128
            i_n_xdl = i_n_block // 64 * 32
            i_n = i_n_xdl + i_n_pos
            i_k = i_thread % 64 // 32 + i_thread // 128 * 2
            i_k_pos = i_k * 8
            for i_vec in range(t_read_vec):
                ds_read_pos[i_thread][i_vec] = lds_pos[i_n][i_k_pos + i_vec]

        self.ds_read_pos = ds_read_pos
        self.ds_read_bank = (ds_read_pos // 2) % 32

        
if __name__ == "__main__":
    k0 = 4
    k1 = 8
    m0 = 16
    m1 = 8

    m1xk1_padding = 0

    t_write_vec = 2
    t_write_len = 8
    t_read_vec = 8
    
    lds_poss = lds_positions(k0, k1, m0, m1, m1xk1_padding, t_write_len, t_write_vec, t_read_vec)
    k = k0 * k1
    lds_pos, lds_bank = lds_poss.compute_lds_bank()
    np.savetxt("lds_pos.txt", lds_pos, "%4d")
    np.savetxt("lds_bank.txt", lds_bank, "%4d")
    np.savetxt("lds_bank.csv", lds_bank, "%d", ";")

    lds_bank_df = lds_poss.gen_pd_dict()
    ds_write_pos = lds_poss.gen_ds_write_pos_bank()
    lds_poss.gen_ds_read_pos()
    np.savetxt("ds_write_pos.txt", ds_write_pos, "%4d")
    lds_poss.gen_ds_bank_dict()
