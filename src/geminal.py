import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk
from typing import Optional

from utils import logdet_matmul

class Geminal(hk.Module):

    def __init__(self, 
                 depth :int,
                 h1_size:int, 
                 h2_size:int, 
                 Nf:int,
                 L:float,
                 K: int = 0,
                 init_stddev:float = 0.01,
                 name: Optional[str] = None
                 ):
        super().__init__(name=name)
        assert (depth >= 2)
        self.depth = depth
        self.h1_size = h1_size
        self.Nf = Nf
        self.L = L
        self.K = K
        self.init_stddev = init_stddev
  
        self.fc_e = [hk.Linear(h1_size, w_init=hk.initializers.TruncatedNormal(self.init_stddev)) for d in range(depth)]
        self.fc_ee = [hk.Linear(h2_size, w_init=hk.initializers.TruncatedNormal(self.init_stddev)) for d in range(depth-1)]
        self.fc_ep = [hk.Linear(h2_size, w_init=hk.initializers.TruncatedNormal(self.init_stddev)) for d in range(depth-1)]
    
    def _ee_feature(self, x):
        n, dim = x.shape[0], x.shape[1]
        rij = (jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))
        
        #|r| calculated with periodic consideration
        r = jnp.linalg.norm(jnp.sin(np.pi*rij/self.L)+jnp.eye(n)[..., None], axis=-1)*(1.0-jnp.eye(n))* (self.L/np.pi)
        
        f = [r[..., None]]
        for n in range(1, self.Nf+1):
            f += [jnp.cos(2*np.pi*rij/self.L*n), jnp.sin(2*np.pi*rij/self.L*n)]
        return jnp.concatenate(f, axis=-1)

    def _ep_feature(self, x, s):
        n, dim = x.shape[0], x.shape[1]
        rij = (jnp.reshape(x, (n, 1, dim)) - jnp.reshape(s, (1, n, dim)))
        
        #|r| calculated with periodic consideration
        r = jnp.linalg.norm(jnp.sin(np.pi*rij/self.L), axis=-1)*(self.L/np.pi)

        f = [r[..., None]]
        for n in range(1, self.Nf+1):
            f += [jnp.cos(2*np.pi*rij/self.L*n), jnp.sin(2*np.pi*rij/self.L*n)]
        return jnp.concatenate(f, axis=-1)

    def _combine(self, e, ee, ep):
        
        n = e.shape[0] # number of electrons

        h1s = jnp.split(e, [n//2], axis=0)
        g1 = [jnp.mean(h, axis=0, keepdims=True) for h in h1s if h.size > 0]
        g1 = [jnp.tile(g, [n, 1]) for g in g1]

        h2s = jnp.split(ee, [n//2], axis=0)
        g2 = [jnp.mean(h, axis=0) for h in h2s if h.size > 0]

        g3 = jnp.mean(ep, axis=1) # mean over proton position

        return jnp.concatenate([e] + g1 + g2 + [g3], axis=1)

    def __call__(self, sx, kpoints):

        n, dim = sx.shape[0], sx.shape[1]
        s, x = jnp.split(sx, [n//2], axis=0)

        e = jnp.repeat(kpoints[0][None, :], n//2, axis=0) # twist as a feature
        ee = self._ee_feature(x) 
        ep = self._ep_feature(x, s)

        for d in range(self.depth-1):

            f = self._combine(e, ee, ep)
            e_update = jnp.tanh(self.fc_e[d](f))
            ee_update = jnp.tanh(self.fc_ee[d](ee))
            ep_update = jnp.tanh(self.fc_ep[d](ep))

            if d > 0:
                e = e_update + e
                ee = ee_update + ee 
                ep = ep_update + ep
            else:
                e = e_update
                ee = ee_update
                ep = ep_update

        f = self._combine(e, ee, ep)
        e = jnp.tanh(self.fc_e[-1](f)) + e

        '''
        #jastrow
        u = hk.get_parameter("u", [self.K, self.h2_size], init=hk.initializers.TruncatedNormal(stddev=self.init_stddev), dtype=x.dtype)
        v = hk.get_parameter("v", [self.K, self.h2_size], init=hk.initializers.TruncatedNormal(stddev=self.init_stddev), dtype=x.dtype)
        jastrow = jnp.einsum("ka,ija->k", u, ee) + jnp.einsum("ka,ija->k", v, ep)
        '''

        #geminal orbital
        orb_fn = hk.Linear(self.h1_size, w_init=hk.initializers.TruncatedNormal(self.init_stddev))
        orb = orb_fn(e.astype(jnp.complex128))

        w = hk.get_parameter("w", [self.K, self.h1_size, self.h1_size], init=hk.initializers.TruncatedNormal(stddev=self.init_stddev), dtype=x.dtype)
        phi = jnp.einsum("ia,kab,jb->kij", orb[:n//4], w, orb[n//4:]) \
             +jnp.ones((n//4,n//4))[None, :, :]

        #plane-wave envelope
        nk = kpoints.shape[0]//2
        backflow = hk.Linear(dim, w_init=hk.initializers.TruncatedNormal(self.init_stddev), with_bias=False)
        z = backflow(e) + x # backflow coordinates
        D_up = 1 / self.L**(dim/2) * jnp.exp(1j * (kpoints[:nk, None, :] * z[None, :n//4, :]).sum(axis=-1))
        D_dn = 1 / self.L**(dim/2) * jnp.exp(1j * (kpoints[nk:, None, :] * z[None, n//4:, :]).sum(axis=-1))
        
        mlp = hk.nets.MLP([self.h1_size, self.K*(nk-1)], w_init=hk.initializers.TruncatedNormal(self.init_stddev), activation=jnp.tanh)
        f = jax.nn.softplus(mlp(kpoints[0])).reshape(self.K, nk-1) # twist dependent momentum occupation
        f = jnp.concatenate([jnp.ones((self.K, 1)), f], axis=1) 

        D = jnp.einsum('ai,ka,aj->kij', D_up, f, jnp.conjugate(D_dn))

        phase, logabsdet = logdet_matmul([D*phi])
        
        return logabsdet + jnp.log(phase)  
