import numpy as np 

def sp_orbitals(dim, Emax=60):
    """
        Compute index (n_1, ..., n_dim) and corresponding energy n_1^2 + ... + n_dim^2
    of all single-particle plane wave in spatial dimension `dim` whose energy
    does not exceed `Emax`.

    OUTPUT SHAPES:
        indices: (n_orbitals, dim), Es: (n_orbitals)
        (n_orbitals stands for total number of single-particle plane wave orbitals
    that fulfil the criteria.)
    """
    n_max = int(np.floor(np.sqrt(Emax)))
    indices = []
    Es = []
    if dim == 2:
        for nx in range(-n_max, n_max+1):
            for ny in range(-n_max, n_max+1):
                E = nx**2 + ny**2
                if E <= Emax:
                    indices.append((nx, ny))
                    Es.append(E)
    elif dim == 3:
        for nx in range(-n_max, n_max+1):
            for ny in range(-n_max, n_max+1):
                for nz in range(-n_max, n_max+1):
                    E = nx**2 + ny**2 + nz**2
                    if E <= Emax:
                        indices.append((nx, ny, nz))
                        Es.append(E)
    indices, Es = np.array(indices), np.array(Es)
    sort_idx = Es.argsort()
    indices, Es = indices[sort_idx], Es[sort_idx]
    return indices, Es

if __name__ == "__main__":
    for dim in (2, 3):
        indices, Es = sp_orbitals(dim)

        print("---- Closed-shell (spinless) electron numbers in dim = %d ----" % dim)
        Ef = Es[0]
        for i in range(Es.size):
            if (Es[i] != Ef):
                print("n = %d, Ef = %d" % (i, Ef))
                Ef = Es[i]
        print("n = %d, Ef = %d" % (Es.size, Es[-1]))

    indices, Es = sp_orbitals(3)
    print (indices[:33])
