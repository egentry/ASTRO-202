import numpy as np 

def integrator(func, a, b, *args, n_steps=10**4, log_spacing=False, **kwargs):
	"""
	integrate function 'func'(x) from x=a to x=b

	Parameters
	----------
	func : callable
		Function to be integrated.
		Expects a function call:
			func(x, *args, **kwargs)
	a : float
		lower limit of integration
	b : float 
		upper limit of integration
	*args : tuple, optional
		additional args to pass to 'func'
	n_steps : int, optional
		number of steps in numeric integration
	log_spacing : bool, optional
		use log-spaced steps rather than lin-spaced steps
	**kwargs : dict, optional
		additional kwargs to pass to 'func'

	Returns
	-------
	integrated : float

	Notes
	-----
	If log_spacing is True, then a and b must be positive
	"""

	if log_spacing:
		xs = np.logspace(np.log10(a), np.log10(b), n_steps+1)
	else:
		xs = np.linspace(a, b, n_steps+1)
	dxs = xs[1:] - xs[:-1]

	integrated = 0
	for x, dx in zip(xs, dxs):
		integrated += dx*func(x, *args, **kwargs)
	return integrated
	