from __future__ import division
import numpy.testing as npt

from integrator import integrator
from planck_physics import B_nu, B

class TestIntegrator:

	def test_integrate_x_squared(self):
		f = lambda x : x**2

		integrated = integrator(f, 0, 1, n_steps=10**5) 
		npt.assert_allclose(integrated, 1/3, rtol=1e-4)

	def test_integrate_x_cubed(self):
		f = lambda x : x**3

		integrated = integrator(f, 0, 1, n_steps=10**5) 
		npt.assert_allclose(integrated, 1/4, rtol=1e-4)

	def test_integrate_planck(self):
		T = 300
		integrated = integrator(B_nu, 1e6, 1e16, T, n_steps=10**5)
		analytic = B(T)

		npt.assert_allclose(integrated, analytic, rtol=1e-4)
