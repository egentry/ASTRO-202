import numpy as np
from astropy import constants as const

h   = const.h.cgs.value
c   = const.c.cgs.value
k_B = const.k_B.cgs.value
sigma = const.sigma_sb.cgs.value

def B_nu(nu, T):
	"""
	Specific brightness

	Parameters
	----------
	nu : float
		frequency [Hz]
	T : float
		Temperature [K]

	Returns
	-------
	B_nu : float
		Specific brightness [ergs s**-1 cm**-2 steradian**-1 Hz**-1]
	"""
	return (2*h*nu**3 / c**2) / (np.exp(h*nu/(k_B*T))-1)

def B(T):
	"""
	Total brightness as a function of temperature (analytic)

	Parameters
	----------
	T : float
		Temperature [K]

	Returns
	-------
	B : float
		Brightness [ergs s**-1 cm**-2 steradian**-1]
	"""
	return sigma * T**4 / np.pi

def d_B_nu_d_nu_dimensionless(x):
	"""
	Calculates d(B_nu) / d (nu), 
	as a function of dimensionless units, x = (h nu / k_B T)

	Parameters
	----------
	x : float

	Returns
	-------
	d_B_nu_d_nu_dimensionless : float
		Not normalized to anything meaningful

	"""
	return (3*x**2 / (np.exp(x)-1)) - (x**3 * np.exp(x) / (np.exp(x)-1)**2)

def d_B_lambda_d_lambda_dimensionless(y):
	"""
	Calculates d(B_lambda) / d (lambda), 
	as a function of dimensionless units, y = (h c / lambda k_B T)

	Parameters
	----------
	y: float

	Returns
	-------
	d_B_lambda_d_lambda_dimensionless : float
		Not normalized to anything meaningful
		
	"""
	return (5*y**4 / (np.exp(y)-1)) - (y**5 * np.exp(y) / (np.exp(y)-1)**2)


def d_B_nu_d_T_dimensionless(x):
	"""
	Calculates d(B_nu) / d (T), 
	as a function of dimensionless units, x = (h nu / k_B T)

	Parameters
	----------
	x : float

	Returns
	-------
	d_B_nu_d_T_dimensionless : float
		Not normalized to anything meaningful

	"""
	return x**4 * np.exp(x) / (np.exp(x)-1)**2

def d_B_nu_d_T_dimensionless(x):
	"""
	Calculates d(B_nu) / d (T), 
	as a function of dimensionless units, x = (h nu / k_B T)

	Parameters
	----------
	x : float

	Returns
	-------
	d_B_nu_d_T_dimensionless : float
		Not normalized to anything meaningful

	"""
	return x**4 * np.exp(x) / (np.exp(x)-1)**2

def d_B_nu_d_T_d_nu_dimensionless(x):
	"""
	Calculates d^2(B_nu) / d (T) / d (nu), 
	as a function of dimensionless units, x = (h nu / k_B T)

	Parameters
	----------
	x : float

	Returns
	-------
	d_B_nu_d_T_d_nu_dimensionless : float
		Not normalized to anything meaningful

	"""
	return - np.exp(x)*x**3 * (np.exp(x)*(x-4)+x+4) / (np.exp(x)-1)**3


