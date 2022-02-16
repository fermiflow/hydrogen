import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk
from typing import Optional

class FermiNet(hk.Module):

    def __init__(self, 
                 depth :int,
                 h1_size:int, 
                 h2_size:int, 
                 Nf:int,
                 L:float,
                 is_wfn:bool,
                 init_stddev:float = 0.01,
                 name: Optional[str] = None
                 ):
        super().__init__(name=name)
        self.depth = depth
        self.Nf = Nf
        self.L = L
        self.is_wfn = is_wfn
        self.init_stddev = init_stddev


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
        r = r[..., None]
        
        f = [r]
        for n in range(1, self.Nf+1):
            f += [jnp.cos(2*np.pi*rij*n/self.L), jnp.sin(2*np.pi*rij*n/self.L)]
        return jnp.concatenate(f, axis=-1)


    def _combine(self, h1, h2):
        n = h1.shape[0]

        partitions = [n//2, n//2+n//4] if self.is_wfn else [n]

        h1s = jnp.split(h1, partitions, axis=0)
        h2s = jnp.split(h2, partitions, axis=0)

        g1 = [jnp.mean(h, axis=0, keepdims=True) for h in h1s if h.size > 0]
        g2 = [jnp.mean(h, axis=0) for h in h2s if h.size > 0]

        g1 = [jnp.tile(g, [n, 1]) for g in g1]

        return jnp.concatenate([h1] + g1 + g2, axis=1) 

    def __call__(self, x):
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

        if self.is_wfn:
            
            #orbital
            w = hk.get_parameter("w", [n//2, h1.shape[-1]], init=hk.initializers.TruncatedNormal(stddev=self.init_stddev))
            b = hk.get_parameter("b", [n//2], init=jnp.zeros)
            phi = w@h1[n//2:].T + b + jnp.ones((n//2,n//2))

            #envlope
            z = final(h1[n//2:]) + x[n//2:] # backflow coordinates
            rpe = jnp.reshape(x[:n//2], (n//2, 1, dim)) - jnp.reshape(z, (1, n//2, dim)) 
            rpe = rpe - self.L*jnp.rint(rpe/self.L)
            r = jnp.linalg.norm(rpe, axis=-1)

            alpha = hk.get_parameter("alpha", [1], init=hk.initializers.Constant(1.4))
            D = jnp.exp(-r*alpha) # e^(-r/a0) = e^(-r*rs) so a good initilization is alpha = rs

            _, logabsdet = jnp.linalg.slogdet(D*phi)

            return logabsdet
        else:
            return final(h1) + x
