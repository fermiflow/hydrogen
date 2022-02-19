import jax
import jax.numpy as jnp

from functools import partial

def make_logpsi(flow, L, rs):

    def logpsi(x, params, s):

        """
            Generic function that computes ln Psi(x) given proton position
        `s`, a set of electron coordinates `x`, and flow parameters `params`.

        INPUT:
            x: (n, dim)     
            s: (n, dim)

        OUTPUT:
            a single complex number ln Psi(x), given in the form of a 2-tuple (real, imag).
        """
        
        n, dim = x.shape
        log_phi = flow.apply(params, None, jnp.concatenate([s, x]))
    
        return jnp.stack([log_phi.real,
                          log_phi.imag])

    return logpsi

def make_logpsi_grad_laplacian(logpsi, forloop=True, hutchinson=False):

    @partial(jax.vmap, in_axes=(0, None, 0), out_axes=0)
    def logpsi_vmapped(x, params, s):
        logpsix = logpsi(x, params, s)
        return logpsix[0] + 1j * logpsix[1]

    @partial(jax.vmap, in_axes=(0, None, 0, None), out_axes=0)
    def logpsi_grad_laplacian(x, params, s, key):
        """
            Computes the gradient and laplacian of logpsi w.r.t. electron coordinates x.
        The final result is in complex form.

        Relevant dimensions: (after vmapped)

        INPUT:
            x: (batch, n, dim)  s: (batch, n, dim)
        OUTPUT:
            grad: (batch, n, dim)   laplacian: (batch,)
        """

        grad = jax.jacrev(logpsi)(x, params, s)
        grad = grad[0] + 1j * grad[1]
        print("Computed gradient.")

        n, dim = x.shape
        x_flatten = x.reshape(-1)
        grad_logpsi = jax.jacrev(lambda x: logpsi(x.reshape(n, dim), params, s))

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

    def logpsi_grad_laplacian_hutchinson(x, params, s, key):

        v = jax.random.normal(key, x.shape)

        @partial(jax.vmap, in_axes=(0, None, 0, 0), out_axes=0)
        def logpsi_grad_random_laplacian(x, params, s, v):
            """
                Compute the laplacian as a random variable `v^T hessian(ln Psi_n(x)) v`
            using the Hutchinson's trick.

                The argument `v` is a random "vector" that has the same shape as `x`,
            i.e., (after vmapped) (batch, n, dim).
            """

            grad, hvp = jax.jvp( jax.jacrev(lambda x: logpsi(x, params, s)),
                                 (x,), (v,) )

            grad = grad[0] + 1j * grad[1]
            print("Computed gradient.")

            random_laplacian = (hvp * v).sum(axis=(-2, -1))
            random_laplacian = random_laplacian[0] + 1j * random_laplacian[1]
            print("Computed Hutchinson's estimator of laplacian.")

            return grad, random_laplacian

        logpsi_grad_laplacian = logpsi_grad_random_laplacian 
        return logpsi_grad_laplacian(x, params, s, v)

    return logpsi_vmapped, \
           (logpsi_grad_laplacian_hutchinson if hutchinson else logpsi_grad_laplacian)

def make_logpsi2(logpsi):

    @partial(jax.vmap, in_axes=(0, None, 0), out_axes=0)
    def logpsi2(x, params, s):
        """ logp = logpsi + logpsi* = 2 Re logpsi """
        return 2 * logpsi(x, params, s)[0]

    return logpsi2

def make_quantum_score(logpsi):

    @partial(jax.vmap, in_axes=(0, None, 0), out_axes=0)
    def quantum_score_fn(x, params, s):
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
        grad_params = jax.jacrev(logpsi, argnums=1)(x, params, s)
        return jax.tree_map(lambda jac: jac[0] + 1j * jac[1], grad_params)

    return quantum_score_fn
