import jax
import jax.numpy as jnp

from functools import partial

def make_logpsi(flow):

    def logpsi(x, params, s, k):

        """
            Generic function that computes ln Psi(x) given momenta `k` and proton position
        `s`, a set of electron coordinates `x`, and flow parameters `params`.

        INPUT:
            x: (n, dim)     
            s: (n, dim)
            k: (nk, dim)

        OUTPUT:
            a single complex number ln Psi(x), given in the form of a 2-tuple (real, imag).
        """
        
        log_phi = flow.apply(params, None, jnp.concatenate([s, x]), k)
    
        return jnp.stack([log_phi.real,
                          log_phi.imag])

    return logpsi

def make_logpsi_grad_laplacian(logpsi, forloop=True):

    @partial(jax.vmap, in_axes=(0, None, 0, 0), out_axes=0) # T
    @partial(jax.vmap, in_axes=(0, None, 0, None), out_axes=0) # W
    @partial(jax.vmap, in_axes=(0, None, None, None), out_axes=0) # B
    def logpsi_vmapped(x, params, s, k):
        logpsix = logpsi(x, params, s, k)
        return logpsix[0] + 1j * logpsix[1]

    @partial(jax.vmap, in_axes=(0, None, 0, 0), out_axes=0) # T
    @partial(jax.vmap, in_axes=(0, None, 0, None), out_axes=0) # W
    @partial(jax.vmap, in_axes=(0, None, None, None), out_axes=0) # B
    def logpsi_grad_laplacian(x, params, s, k):
        """
            Computes the gradient and laplacian of logpsi w.r.t. electron coordinates x.
        The final result is in complex form.

        Relevant dimensions: (after vmapped)

        INPUT:
            x: (T, W, B, n, dim)  
            s: (T, W, n, dim)
            k: (T, nk, dim)
        OUTPUT:
            grad: (T, W, B, n, dim)   laplacian: (T, W, B,)
        """

        grad = jax.jacrev(logpsi)(x, params, s, k)
        grad = grad[0] + 1j * grad[1]
        print("Computed gradient.")

        n, dim = x.shape
        x_flatten = x.reshape(-1)
        grad_logpsi = jax.jacrev(lambda x: logpsi(x.reshape(n, dim), params, s, k))

        def _laplacian(x):
            if forloop:
                print("forloop version...")
                def body_fun(i, val):
                    _, tangent = jax.jvp(grad_logpsi, (x,), (eye[i],))
                    return val + tangent[0, i] + 1j * tangent[1, i]
                eye = jnp.eye(x.shape[0])
                laplacian = jax.lax.fori_loop(0, x.shape[0], body_fun, 0.+0.j)
            else:
                print("vmap version...")
                def body_fun(x, basevec):
                    _, tangent = jax.jvp(grad_logpsi, (x,), (basevec,))
                    return (tangent * basevec).sum(axis=-1)
                eye = jnp.eye(x.shape[0])
                laplacian = jax.vmap(body_fun, (None, 1), 1)(x, eye).sum(axis=-1)
                laplacian = laplacian[0] + 1j * laplacian[1]
            return laplacian

        laplacian = _laplacian(x_flatten)
        print("Computed laplacian.")

        return grad, laplacian

    return logpsi_vmapped, logpsi_grad_laplacian

def make_logpsi2(logpsi):
    
    @partial(jax.vmap, in_axes=(0, None, 0, 0), out_axes=0) # T
    @partial(jax.vmap, in_axes=(0, None, 0, None), out_axes=0) # W
    @partial(jax.vmap, in_axes=(0, None, None, None), out_axes=0) # B
    def logpsi2(x, params, s, k):
        """ logp = logpsi + logpsi* = 2 Re logpsi 
        x: (T, W, B, n, dim)
        s: (T, W, n, dim)
        k: (T, nk, dim)
        """
        return 2 * logpsi(x, params, s, k)[0]
    return logpsi2

def make_quantum_score(logpsi):

    @partial(jax.vmap, in_axes=(0, None, 0, 0), out_axes=0) # T
    @partial(jax.vmap, in_axes=(0, None, 0, None), out_axes=0) # W
    @partial(jax.vmap, in_axes=(0, None, None, None), out_axes=0) # B
    def quantum_score_fn(x, params, s, k):
        """
            Computes the "quantum score function", i.e., the gradient of ln Psi_n(x)
        w.r.t. the flow parameters.
            This function can be useful for stochastic reconfiguraton, the
        second-order optimization algorithm based on quantum (as well as classical)
        Fisher information matrix.

        Relevant dimension: (after vmapped)

        OUTPUT:
            a pytree of the same structure as `params`, in which each leaf node has
        an additional leading batch dimension.
        """
        grad_params = jax.jacrev(logpsi, argnums=1)(x, params, s, k)
        return jax.tree_map(lambda jac: jac[0] + 1j * jac[1], grad_params)

    return quantum_score_fn
