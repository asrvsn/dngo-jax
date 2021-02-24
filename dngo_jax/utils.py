'''
Jax utils
'''

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp

def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""
    return array.reshape(array.shape[1]//nrows, nrows, -1, ncols).swapaxes(1, 2)

def normalize(X):
	''' Normalize batched inputs ''' 
	X_mu, X_std = jnp.mean(X, axis=0), jnp.std(X, axis=0)
	X_bar = (X - X_mu) / X_std
	return X_bar, X_mu, X_std

def unnormalize(X_bar, X_mu, X_std):
	return X_bar*X_std + X_mu

def mse_loss(X1, X2):
	def squared_error(x1, x2):
		return (x1-x2) @ (x1-x2) / 2
	return jnp.mean(jax.vmap(squared_error)(X1, X2), axis=0)