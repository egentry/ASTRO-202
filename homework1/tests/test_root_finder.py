import numpy.testing as npt

from root_finder import root_finder
from planck_physics import 	d_B_nu_d_nu_dimensionless, \
							d_B_lambda_d_lambda_dimensionless

class TestRootFinder:

	def test_finding_cubic_root(self):
		f = lambda x : x**3 + 8

		root = root_finder(f, 1, max_evals=10**2) 
		npt.assert_allclose(root, -2, rtol=1e-4)

	def test_finding_Wein_peak_lambda(self):
		root = root_finder(d_B_lambda_d_lambda_dimensionless,
						   5, max_evals=10**2) 
		npt.assert_allclose(root, 4.96511, rtol=1e-4)

	def test_finding_Wein_peak_nu(self):
		root = root_finder(d_B_nu_d_nu_dimensionless,
						   3, max_evals=10**2) 
		npt.assert_allclose(root, 2.82144, rtol=1e-4)
