import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk
from typing import Optional

from utils import logdet_matmul

class FermiNet(hk.Module):

    def __init__(self, 
                 depth :int,
                 h1_size:int, 
                 h2_size:int, 
                 Nf:int,
                 L:float,
                 K: int = 0,
                 init_stddev:float = 0.01,
                 rs: Optional[float] = 1.4,
                 name: Optional[str] = None
                 ):
        super().__init__(name=name)
        self.depth = depth
        self.Nf = Nf
        self.L = L
        self.K = K
        self.init_stddev = init_stddev
        self.rs = rs
  
        self.fc1 = [hk.Linear(h1_size, w_init=hk.initializers.TruncatedNormal(self.init_stddev)) for d in range(depth)]
        self.fc2 = [hk.Linear(h2_size, w_init=hk.initializers.TruncatedNormal(self.init_stddev)) for d in range(depth-1)]
    
    def _h1(self, x):
        return jnp.zeros_like(x)
        #f = []
        #for n in range(1, self.Nf+1):
        #    f += [jnp.cos(2*np.pi*x*n/self.L), jnp.sin(2*np.pi*x*n/self.L)]
        #return jnp.concatenate(f, axis=-1)

    def _h2(self, x):
        n, dim = x.shape[0], x.shape[1]
        rij = (jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))
        
        #|r| calculated with periodic consideration
        r = jnp.linalg.norm(jnp.sin(2*np.pi*rij/self.L)+jnp.eye(n)[..., None], axis=-1) *(1.0-jnp.eye(n))
        
        f = [r[..., None]]
        for n in range(1, self.Nf+1):
            f += [jnp.cos(2*np.pi*rij*n/self.L), jnp.sin(2*np.pi*rij*n/self.L)]
        return jnp.concatenate(f, axis=-1)


    def _combine(self, h1, h2):
        n = h1.shape[0]

        partitions = [n//2, n//2+n//4] if self.K >0 else [n]

        h1s = jnp.split(h1, partitions, axis=0)
        h2s = jnp.split(h2, partitions, axis=0)

        g1 = [jnp.mean(h, axis=0, keepdims=True) for h in h1s if h.size > 0]
        g2 = [jnp.mean(h, axis=0) for h in h2s if h.size > 0]

        g1 = [jnp.tile(g, [n, 1]) for g in g1]

        return jnp.concatenate([h1] + g1 + g2, axis=1) 

    def __call__(self, x, kpoints=None):

        n, dim = x.shape[0], x.shape[1]

        h1 = self._h1(x) 
        h2 = self._h2(x)

        for d in range(self.depth-1):
            f = self._combine(h1, h2)
            h1_update = jnp.tanh(self.fc1[d](f))
            h2_update = jnp.tanh(self.fc2[d](h2))

            if d > 0:
                h1 = h1_update + h1
                h2 = h2_update + h2
            else:
                h1 = h1_update 
                h2 = h2_update

        f = self._combine(h1, h2)
        h1 = jnp.tanh(self.fc1[-1](f)) + h1

        final = hk.Linear(dim, w_init=hk.initializers.TruncatedNormal(self.init_stddev))

        if self.K > 0:

            #ep jastrow
            rij = jnp.reshape(x[:n//2], (n//2, 1, dim)) - jnp.reshape(x[n//2:], (1, n//2, dim)) 
            r = jnp.linalg.norm(jnp.sin(2*jnp.pi*rij/self.L), axis=-1)*(self.L/(2*jnp.pi))
            alpha = hk.get_parameter("alpha", [self.K], init=hk.initializers.TruncatedNormal(stddev=self.init_stddev))
            jastrow_ep = jnp.exp(-jnp.sum(self.rs*r/(1 + jax.nn.softplus(alpha)[:, None, None] * r), axis=(1,2)))

            #ee jastrow for unlike spins
            rij = jnp.reshape(x[n//2:n//2+n//4], (n//4, 1, dim)) - jnp.reshape(x[n//2+n//4:], (1, n//4, dim)) 
            r = jnp.linalg.norm(jnp.sin(2*jnp.pi*rij/self.L), axis=-1)*(self.L/(2*jnp.pi))
            beta = hk.get_parameter("beta", [self.K], init=hk.initializers.TruncatedNormal(stddev=self.init_stddev))
            jastrow_ee = jnp.exp(jnp.sum(0.5*self.rs*r/(1 + jax.nn.softplus(beta)[:, None, None] * r), axis=(1,2)))

            #orbital
            w = hk.get_parameter("w", [self.K, h1.shape[-1], h1.shape[-1]], init=hk.initializers.TruncatedNormal(stddev=self.init_stddev))
            b = hk.get_parameter("b", [self.K, h1.shape[-1]], init=jnp.zeros)
            c = hk.get_parameter("c", [self.K, h1.shape[-1]], init=jnp.zeros)

            phi = jnp.einsum("ia,kab,jb->kij", h1[n//2:n//2+n//4], w, h1[n//2+n//4:]) \
                 +jnp.einsum("ia,ka->ki", h1[n//2:n//2+n//4], b)[:, :, None] \
                 +jnp.einsum("ka,ia->ki", c, h1[n//2+n//4:])[:, None, :] \
                 +jnp.ones((n//4,n//4))[None, :, :]

            #geminal envelope
            nk = kpoints.shape[0]//2
            z = final(h1[n//2:]) + x[n//2:] # backflow coordinates
            D_up = 1 / self.L**(dim/2) * jnp.exp(1j * (kpoints[:nk, None, :] * z[None, :n//4, :]).sum(axis=-1))
            D_dn = 1 / self.L**(dim/2) * jnp.exp(1j * (kpoints[nk:, None, :] * z[None, n//4:, :]).sum(axis=-1))
            
            f = hk.get_parameter("f", [self.K, nk], init=hk.initializers.TruncatedNormal(stddev=self.init_stddev))
            f = f + jnp.concatenate([jnp.ones(n//4), jnp.zeros(nk-n//4)]) # here we use diagonal f to ensure continuous translation symm
            D = jnp.einsum("ai,ka,aj->kij", D_up, f, jnp.conjugate(D_dn))
        
            phase, logabsdet = logdet_matmul([D*phi], jastrow_ep * jastrow_ee)
            
            return logabsdet + jnp.log(phase)  
        else:
            return final(h1) + x
