from typing import Tuple

import numpy as np
from pyscf.pbc import dft, gto, scf

# We choose energy unit to be Rydberg. 1 Hartree = 2 Ry.
Ry = 2.

class Hydrogen:
    def __init__(self, length: float, position_nuc: np.ndarray) -> None:
        """
        Args:
            length: length of hydrogen cell (unit: Bohr).
            position_nuc: (n_nuc, 3) array, positions of hydrogen nuclei.
        """

        self.n_H = position_nuc.shape[0]
        self.n_alpha = self.n_beta = self.n_H // 2
        self.cell = gto.Cell()
        self.cell.unit = 'B'
        self.cell.atom = []
        for ie in range(self.n_H):
            self.cell.atom.append(['H', tuple(position_nuc[ie])])
        self.cell.spin = position_nuc.shape[0] % 2
        self.cell.basis = 'sto-3g'
        self.cell.a = np.eye(3) * length
        self.cell.precision = 1e-5
        self.cell.build()

        # gamma point [0, 0, 0] (unit: reciprocal lattice vector, 2*pi/length)
        # First Brillouin zone: [-0.5, 0.5]^3
        # baldereschi K point [0.25, 0.25, 0.25]
        self.kpts = self.cell.make_kpts([1, 1, 1], scaled_center=[0., 0., 0.])
        self.kmf = scf.khf.KRHF(self.cell, kpts=self.kpts)
        self.kmf.conv_tol = 1e-6
        self.kmf.verbose = 0
        self.kmf.kernel()
        self.mo_coeff = self.kmf.mo_coeff[0]  # (n_ao, n_mo)
        self.mo_coeff_alpha = self.mo_coeff[..., 0:self.n_alpha]  # (n_ao, n_alpha)
        self.mo_coeff_beta = self.mo_coeff[..., 0:self.n_beta]  # (n_ao, n_beta)


    def E(self) -> float:
        """ Total energy = K + Vpp + Vee + Vep. """
        return Ry * self.kmf.e_tot

    def K(self) -> float:
        self.t_matrix = scf.hf.get_t(self.cell, kpt=self.kpts)[0]
        t_diag_alpha = np.einsum('ji,jk,km->im', self.mo_coeff_alpha.conj(), self.t_matrix, self.mo_coeff_alpha).real
        t_diag_beta = np.einsum('ji,jk,km->im', self.mo_coeff_beta.conj(), self.t_matrix, self.mo_coeff_beta).real
        return Ry * (np.trace(t_diag_alpha) + np.trace(t_diag_beta))

    def Vpp(self) -> float:
        return Ry * self.kmf.energy_nuc()

    def Vee(self) -> float:
        return Ry * self.kmf.energy_elec()[1]

    def Vep(self) -> float:
        self.p_matrix = scf.hf.get_pp(self.cell, kpt=self.kpts[0])
        p_diag_alpha = np.einsum('ji,jk,km->im', self.mo_coeff_alpha.conj(), self.p_matrix, self.mo_coeff_alpha).real
        p_diag_beta = np.einsum('ji,jk,km->im', self.mo_coeff_beta.conj(), self.p_matrix, self.mo_coeff_beta).real
        return Ry * (np.trace(p_diag_alpha) + np.trace(p_diag_beta))

    def E_elec(self) -> float:
        """ K + Vep + Vee. """
        return Ry * self.kmf.energy_elec()[0]


    def eval_orbitals(self, position_elec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
            Evaluates SCF orbitals from PySCF at a batch of electron coordinates.

        Args:
            position_elec: (batch, n_elec, 3)

        Returns:
            A tuple (slater_matrix_alpha, slater_matrix_beta) of Slater matrices for
            spin-up and spin-down electrons, with shapes (batch, n_alpha, n_alpha)
            and (batch, n_beta, n_beta) respectively. 
        """
        batch = position_elec.shape[0] # batch
        position_elec = np.reshape(position_elec, [-1, 3]) # (batch * n_elec, 3)
        ao_value = self.cell.pbc_eval_ao("GTOval_sph", position_elec, kpts=self.kpts)[0] # (batch * n_elec, n_ao)
        ao_value = np.reshape(ao_value, (batch, self.n_H, -1)) # (batch, n_elec, n_ao)
        ao_value_alpha = ao_value[..., :self.n_alpha, :] # (batch, n_alpha, n_ao)
        ao_value_beta = ao_value[..., self.n_alpha:, :] # (batch, n_beta, n_ao)
        slater_matrix_alpha = np.einsum('ijk,kl->ijl', ao_value_alpha, self.mo_coeff_alpha) # (batch, n_alpha, n_alpha)
        slater_matrix_beta = np.einsum('ijk,kl->ijl', ao_value_beta, self.mo_coeff_beta)  # (batch, n_beta, n_beta)
        return slater_matrix_alpha, slater_matrix_beta

    def geminal(self, position_elec: np.ndarray) -> np.ndarray:
        slater_matrix_alpha, slater_matrix_beta = self.eval_orbitals(position_elec)
        return np.matmul(slater_matrix_alpha, slater_matrix_beta.transpose((0, 2, 1))) + 0J # make it complex in general

    def logpsi(self, position_elec: np.ndarray) -> np.ndarray:
        """ logψ, which is generally complex.

        Args:
            position_elec: (batch, n_elec, 3)
        Returns:
            log_slater_determinant: (batch,)
        """
        slater_matrix_alpha, slater_matrix_beta = self.eval_orbitals(position_elec)
        sign_alpha, logabsdet_alpha = np.linalg.slogdet(slater_matrix_alpha)
        sign_beta, logabsdet_beta = np.linalg.slogdet(slater_matrix_beta)
        sign = sign_alpha * sign_beta
        logabsdet = logabsdet_alpha + logabsdet_beta
        return np.log(sign + 0.j) + logabsdet
        
    logp = lambda self, position_elec: 2. * self.logpsi(position_elec).real

    def logpsi_gradient(self, position_elec: np.ndarray) -> np.ndarray:
        """ ∇ψ/ψ

        Args:
            position_elec: (batch, n_elec, 3)
        Returns:
            logpsi_gradient: (batch, n_elec, 3)
            [[∂_X1ψ, ∂_y1ψ, ∂_z1ψ],
               ...    ...    ...   
             [∂_Xnψ, ∂_ynψ, ∂_znψ]]
        """
        slater_matrix_alpha, slater_matrix_beta = self.eval_orbitals(position_elec)
        slater_matrix_alpha_inv = np.linalg.inv(slater_matrix_alpha) # (batch, n_alpha, n_alpha)
        slater_matrix_beta_inv = np.linalg.inv(slater_matrix_beta) # (batch, n_beta, n_beta)
        batch = position_elec.shape[0] # batch
        position_elec = np.reshape(position_elec, [-1, 3]) # (batch * n_elec, 3)
        ao_gradient = dft.numint.eval_ao(self.cell, position_elec, kpt=self.kpts, deriv=1)[1:4]  # (3, batch * n_elec, n_ao)
        ao_gradient = np.reshape(ao_gradient, (3, batch, self.n_H, -1)) # (3, batch, n_elec, n_ao) leading dim 3 means: ∂x, ∂y, ∂z
        ao_gradient_alpha = ao_gradient[..., 0:self.n_alpha, :]  # (3, batch, n_alpha, n_ao)
        ao_gradient_beta = ao_gradient[..., self.n_alpha:, :]  # (3, batch, n_beta, n_ao)
        mo_gradient_alpha = np.einsum('ibjk,kl->bijl', ao_gradient_alpha, self.mo_coeff_alpha)  # (batch, 3, n_alpha, n_alpha)
        mo_gradient_beta = np.einsum('ibjk,kl->bijl', ao_gradient_beta, self.mo_coeff_beta)  # (batch, 3, n_beta, n_beta) 
        logpsi_gradient_alpha = np.einsum('bijk,bkj->bji', mo_gradient_alpha, slater_matrix_alpha_inv)  # (batch, n_alpha, 3)
        logpsi_gradient_beta = np.einsum('bijk,bkj->bji', mo_gradient_beta, slater_matrix_beta_inv)  # (batch, n_beta, 3)
        logpsi_gradient = np.append(logpsi_gradient_alpha, logpsi_gradient_beta, axis=1)  # (batch, n_elec, 3)
        return logpsi_gradient

    def logpsi_laplacian(self, position_elec: np.ndarray) -> np.ndarray:
        """ ∇^2ψ/ψ

        Args:
            position_elec: (batch, n_elec, 3)
        Returns:
            logpsi_laplacian: (batch,)
        """
        slater_matrix_alpha, slater_matrix_beta = self.eval_orbitals(position_elec)
        slater_matrix_alpha_inv = np.linalg.inv(slater_matrix_alpha) # (batch, n_alpha, n_alpha)
        slater_matrix_beta_inv = np.linalg.inv(slater_matrix_beta) # (batch, n_beta, n_beta)
        batch = position_elec.shape[0] # batch
        position_elec = np.reshape(position_elec, [-1, 3]) # (batch * n_elec, 3)
        ao = dft.numint.eval_ao(self.cell, position_elec, kpt=self.kpts, deriv=2)  # (10, batch * n_elec, n_ao)
        ao_laplacian = np.array([ao[4], ao[7], ao[9]])  # (3, batch * n_elec, n_ao), leading dim 3 means: ∂xx, ∂yy, ∂zz
        ao_laplacian = np.reshape(ao_laplacian, (3, batch, self.n_H, -1)) # (3, batch, n_elec, n_ao)
        ao_laplacian_alpha = ao_laplacian[..., 0:self.n_alpha, :]  # (3, batch, n_alpha, n_ao)
        ao_laplacian_beta = ao_laplacian[..., self.n_alpha:, :]  # (3, batch, n_beta, n_ao)
        mo_laplacian_alpha = np.einsum('ibjk,kl->bjl', ao_laplacian_alpha, self.mo_coeff_alpha)  # (batch, n_alpha, n_alpha)
        mo_laplacian_beta = np.einsum('ibjk,kl->bjl', ao_laplacian_beta, self.mo_coeff_beta)  # (batch, n_beta, n_beta)
        logpsi_laplacian_alpha = np.einsum('bij,bji->b', mo_laplacian_alpha, slater_matrix_alpha_inv) # (batch)
        logpsi_laplacian_beta = np.einsum('bij,bji->b', mo_laplacian_beta, slater_matrix_beta_inv) # (batch)
        logpsi_laplacian = logpsi_laplacian_alpha + logpsi_laplacian_beta # (batch)
        return logpsi_laplacian

    def nuc_grad(self) -> np.ndarray:
        """
            Returns grad of nuclei (n_nuc, 3).
        """
        return self.kmf.run().nuc_grad_method().grad_nuc()

    def density(self, position: np.ndarray) -> np.ndarray:
        """ ρ(r)
            Returns density of electrons
        Args:
            position: (batch, 3)
        Returns:
            rho: (batch,)
        """
        dm = self.kmf.make_rdm1()[0] # (n_ao, n_ao)
        ao_value = self.cell.pbc_eval_ao("GTOval_sph", position, kpts=self.kpts)[0] # (batch, n_ao)
        return np.einsum('bi,bj,ij->b', ao_value, ao_value, dm)

    def gencubefile(self):
        from pyscf.tools import cubegen
        cubegen.density(self.cell, 'cell_den.cube', self.kmf.make_rdm1())


if __name__=='__main__':
    L = 20.0
    d = 1.4
    n = 2
    center = np.array([L/2, L/2, L/2])
    offset = np.array([[d/2, 0., 0.],
                      [-d/2, 0., 0.]])
    x = center + offset
    
    hf = Hydrogen(L, x)

    print ('K', hf.K()/n)
    print ('Vep', hf.Vep()/n)
    print ('Vee', hf.Vee()/n)
    print ('Vpp', hf.Vpp()/n)
    print ('E', hf.E()/n)
