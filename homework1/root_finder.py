import numpy as np
import warnings

def root_finder(func, x0, *args, max_evals=10**4, 
								 tolerance=1e-4,
								 step_limiter=.1,
								 **kwargs):
	"""
	Find roots of 'func', with an initial guess at x0

	Parameters
	----------
	func : callable
		Function to be integrated.
		Expects a function call:
			func(x, *args, **kwargs)
	x0 : float
		initial guess at the root of 'func'
	*args : tuple, optional
		additional args to pass to 'func'
	max_evals : int, optional
		max number of function evaluations allowed
	tolerance : float, optional
		considers x a root if abs(func(x)) < tolerance
	step_limiter : float, optional
		Limit the Newton-Raphson steps from dx to step_limiter*dx
	**kwargs : dict, optional
		additional kwargs to pass to 'func'

	Returns
	-------
	x_next : float
		a root of 'func'
	"""

	x_prev = x0
	x_next = 1.01 * (x_prev + 1e-5)

	y_prev = func(x_prev, *args, **kwargs)
	y_next = func(x_next, *args, **kwargs)

	for i in range(max_evals):
		dy_dx = (y_next - y_prev) / (x_next - x_prev)

		x_prev = x_next
		x_next = x_prev - step_limiter*(y_prev / dy_dx)

		y_prev = y_next
		y_next = func(x_next, *args, **kwargs)
		if np.abs(y_next) < tolerance:
			return x_next

	warnings.warn("root_finder exceeded max_evals", RuntimeWarning)
	return x_next

