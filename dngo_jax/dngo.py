''' 
Neural Bayesian optimization module 
''' 

import numpy as np
import jax
import jax.nn.initializers as initializers
import jax.numpy as jnp
import jax.scipy as jsp
import flax
import flax.linen as nn
import pdb

from dngo_jax.utils import *

class Acquisition(nn.Module):
	'''
	Surrogate cost function taking AdjacencyOp parameters to a real value. 
	Architecture:
	1. Dense layers
	2. Bayesian linear regressor (BLR) layer
		- Trained initially as a dense (linear regression) layer
		- Learned basis functions parametrize BLR
	3. Expected improvement (or perhaps UCB)
		- use MCMC EI to calculated expected improvement
	'''
	alpha_init: float=1.0
	beta_init: float=1000
	hidden_features: int=100
	blr_features: int=32

	def setup(self):
		self.alpha = self.param('alpha', lambda key, shape: self.alpha_init, ())
		self.beta = self.param('beta', lambda key, shape: self.beta_init, ())
		self.fc1 = nn.Dense(features=self.hidden_features)
		self.fc2 = nn.Dense(features=self.blr_features)
		self.basis = lambda x: self.fc2(jnp.tanh(self.fc1(x)))
		self.final = nn.Dense(features=1)

	def __call__(self, x):
		''' Returns the predicted objective ''' 
		x = self.basis(x)
		x = self.final(x)
		return x

	def mll(self, X_bar, Y_bar):
		''' Marginal log-likelihood given normalized training data ''' 
		Phi = self.basis(X_bar)
		K = self.beta*Phi.T@Phi + self.alpha*jnp.eye(Phi.shape[1])
		M = self.beta*jnp.linalg.inv(K)@Phi.T@Y_bar
		n, d = self.X_bar.shape
		mll = (d/2.)*jnp.log(alpha) + \
				(n/2.)*jnp.log(beta) - \
				(n/2.) * jnp.log(2*jnp.pi) - \
				(beta/2.) * jnp.linalg.norm(Y_bar - Phi@M, 2) - \
				(alpha/2.) * M@M - \
				(1/2.) * jnp.slogdet(K)[1]
		return mll, K, M

	def blr_predict(self, X_bar, K, M):
		''' Predicted mean and variance given normalized testing data'''
		Phi = self.basis(X_bar)
		mu = M.T@Phi
		var = Phi.T@jnp.linalg.inv(K)@Phi
		return mu, var

	def EI(self, X_bar, K, M, y_best):
		''' Expected-improvement acquisition value ''' 
		mu, var = self.blr_predict(X_bar, K, M)
		std = jnp.sqrt(var)
		gamma = (y_best - mu) / std
		ei = std * (gamma * jsp.stats.norm.cdf(gamma) + jsp.stats.norm.pdf(gamma))
		return ei

	def UCB(self, X_bar, K, M, beta=2):
		''' Upper confidence bound acquisition value '''
		mu, var = self.blr_predict(X_bar, K, M)
		return mu + beta*jnp.sqrt(var)


# @jax.jit
def train(optim, acq, X_bar, Y_bar, n_epochs):
	''' 
	Train the acquisition function.
	'''
	def loss_fn(params):
		Y_pred = acq.apply({'params': params}, X_bar)
		return mse_loss(Y_pred, Y_bar)
	for _ in range(n_epochs):
		l, grad = jax.value_and_grad(loss_fn, has_aux=True)(optim.target)
		optim = optim.apply_gradient(grad)
	def hyper_loss_fn(theta):
		mll, K, M = acq.apply({'params': {'alpha': theta[0]}, 'beta': theta[1]}, X_bar, Y_bar, method=acq.mll)
		return -mll
	result = jsp.optimize.minimize(
		hyper_loss_fn, 
		jnp.array([optim.target['alpha'], optim.target['beta']]), 
		method='BFGS',
	)
	optim.target['alpha'], optim.target['beta'] = result.x
	return optim

@jax.jit
def maximize(acq, X_bar):
	'''
	Maximize trained acquisition function.
	Incorporates an additional cost component for L1 norm of scale parameter (preferring smaller graphs).
	'''
	pass

if __name__ == '__main__':
	# Test on sinc function

	learning_rate = 0.1
	beta = 0.9
	n_epochs = 10
	X_train = jnp.linspace(0, 1, 100)[:, jnp.newaxis]
	Y_train = jnp.sinc(X_train)
	X_bar, X_mu, X_std = normalize(X_train)
	Y_bar, Y_mu, Y_std = normalize(Y_train)

	key = jax.random.PRNGKey(1)
	acq = Acquisition()
	params = acq.init(key, jnp.ones_like(X_train))['params']
	optim_def = flax.optim.Adam(learning_rate=learning_rate, beta1=beta)
	optim = optim_def.create(params)
	optim = train(optim, acq, X_bar, Y_bar, n_epochs)
	mll, K, M = acq.apply({'params': optim.target}, X_bar, Y_bar, acq.mll)

	X_test = jnp.linspace(0, 1, 14)[:, jnp.newaxis]
	Y_test = jnp.sinc(X_test)
	X_test_bar = (X_test - X_mu) / X_std
	mu_, var_ = acq.apply({'params': optim.target}, X_test_bar, K, M)
	mu = mu_ * Y_std + Y_mu
	var = var_ * (Y_std ** 2)

	plt.plot(X_train.T[0], Y_train.T[0], "ro")
	plt.plot(X_test.T[0], Y_test.T[0], "k--")
	plt.plot(X_test.T[0], mu, "blue")
	plt.fill_between(X_test.T[0], mu + jnp.sqrt(var), mu - jnp.sqrt(var), color="orange", alpha=0.4)
	plt.grid()
	plt.show()