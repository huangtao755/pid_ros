import numpy as np
from typing import Union


class neso:
	def __init__(self,
				 l1: Union[np.ndarray, list],
				 l2: Union[np.ndarray, list],
				 l3: Union[np.ndarray, list],
				 r: Union[np.ndarray, list],
				 k1: Union[np.ndarray, list],
				 k2: Union[np.ndarray, list],
				 dim: int,
				 dt: float):
		self.l1 = np.array(l1)
		self.l2 = np.array(l2)
		self.l3 = np.array(l3)
		self.r = np.array(r)
		self.k1 = np.array(k1)
		self.k2 = np.array(k2)

		self.dt = dt
		self.dim = dim

		self.z1 = np.zeros(dim)
		self.z2 = np.zeros(dim)
		self.z3 = np.zeros(dim)  # 这个是输出
		self.xi = np.zeros(dim)
		self.dot_z1 = np.zeros(dim)
		self.dot_z2 = np.zeros(dim)
		self.dot_z3 = np.zeros(dim)

	def fal(self, xi: np.ndarray):
		res = []
		for i in range(self.dim):
			if np.fabs(xi[i]) <= self.k2[i]:
				res.append(xi[i] / (self.k2[i] ** (1 - self.k1[i])))
			else:
				res.append(np.fabs(xi[i]) ** self.k1[i] * np.sign(xi[i]))
		return np.array(res)

	def set_init(self, x0: np.ndarray, dx0: np.ndarray, syst_dynamic: np.ndarray):
		self.z1 = x0  # 估计x
		self.z2 = dx0  # 估计dx
		self.z3 = np.zeros(self.dim)  # 估计干扰
		self.xi = x0 - self.z1
		self.dot_z1 = self.z2 + self.l1 / self.r * self.fal(xi=self.r ** 2 * self.xi)
		self.dot_z2 = self.z3 + self.l2 * self.fal(xi=self.r ** 2 * self.xi) + syst_dynamic
		self.dot_z3 = self.r * self.l3 * self.fal(xi=self.r ** 2 * self.xi)

	def observe(self, x: np.ndarray, syst_dynamic: np.ndarray):
		self.xi = x - self.z1
		self.dot_z1 = self.z2 + self.l1 / self.r * self.fal(xi=self.r ** 2 * self.xi)
		self.dot_z2 = self.z3 + self.l2 * self.fal(xi=self.r ** 2 * self.xi) + syst_dynamic
		self.dot_z3 = self.r * self.l3 * self.fal(xi=self.r ** 2 * self.xi)
		self.z1 = self.z1 + self.dot_z1 * self.dt  # 观测 pos
		self.z2 = self.z2 + self.dot_z2 * self.dt  # 观测 vel
		self.z3 = self.z3 + self.dot_z3 * self.dt  # 观测 delta
		delta_obs = self.z3.copy()
		dot_delta_obs = self.dot_z3.copy()
		return delta_obs, dot_delta_obs


class robust_differentiator_3rd:
	def __init__(self,
				 m1: Union[np.ndarray, list] = np.array([0, 0, 0]),
				 m2: Union[np.ndarray, list] = np.array([0, 0, 0]),
				 m3: Union[np.ndarray, list] = np.array([0, 0, 0]),
				 n1: Union[np.ndarray, list] = np.array([0, 0, 0]),
				 n2: Union[np.ndarray, list] = np.array([0, 0, 0]),
				 n3: Union[np.ndarray, list] = np.array([0, 0, 0]),
				 use_freq: bool = False,
				 omega: Union[np.ndarray, list] = np.atleast_2d([0, 0, 0]),
				 dim: int = 3,
				 dt: float = 0.001):
		self.m1 = np.array(m1)
		self.m2 = np.array(m2)
		self.m3 = np.array(m3)
		self.n1 = np.array(n1)
		self.n2 = np.array(n2)
		self.n3 = np.array(n3)

		self.m10 = np.zeros(dim)
		self.m20 = np.zeros(dim)
		self.m30 = np.zeros(dim)
		self.n10 = np.zeros(dim)
		self.n20 = np.zeros(dim)
		self.n30 = np.zeros(dim)

		self.a1 = 3. / 4.
		self.a2 = 2. / 4.
		self.a3 = 1. / 4.
		self.b1 = 5. / 4.
		self.b2 = 6. / 4.
		self.b3 = 7. / 4.
		if use_freq:
			for i in range(dim):
				_omega = omega[i]
				m1n1 = _omega[0] + _omega[1] + _omega[2]
				m2n2 = _omega[0] * _omega[1] + _omega[0] * _omega[2] + _omega[1] * _omega[2]
				m3n3 = _omega[0] * _omega[1] * _omega[2]
				self.m10[i] = m1n1
				self.m20[i] = m2n2
				self.m30[i] = m3n3
				self.n10[i] = m1n1
				self.n20[i] = m2n2
				self.n30[i] = m3n3
		else:
			self.m1 = np.array(m1)
			self.m2 = np.array(m2)
			self.m3 = np.array(m3)
			self.n1 = np.array(n1)
			self.n2 = np.array(n2)
			self.n3 = np.array(n3)

		self.z1 = np.zeros(dim)
		self.z2 = np.zeros(dim)
		self.z3 = np.zeros(dim)
		self.dz1 = np.zeros(dim)
		self.dz2 = np.zeros(dim)
		self.dz3 = np.zeros(dim)
		self.dim = dim
		self.dt = dt

		self.threshold = np.array([0.001, 0.001, 0.001])

	def set_init(self, e0: Union[np.ndarray, list], de0: Union[np.ndarray, list], syst_dynamic: Union[np.ndarray, list]):
		self.z1 = np.array(e0)
		self.z2 = np.array(de0)
		self.z3 = np.zeros(self.dim)

		self.dz1 = self.z2.copy()
		self.dz2 = self.z3.copy() + syst_dynamic
		self.dz3 = np.zeros(self.dim)

	@staticmethod
	def sig(x: Union[np.ndarray, list], a):
		return np.fabs(x) ** a * np.sign(x)

	def fal(self, xi: Union[np.ndarray, list], a):
		res = []
		for i in range(self.dim):
			if np.fabs(xi[i]) <= self.threshold[i]:
				res.append(xi[i] / (self.threshold[i] ** np.fabs(1 - a)))
			else:
				res.append(np.fabs(xi[i]) ** a * np.sign(xi[i]))
		return np.array(res)

	def observe(self, syst_dynamic: Union[np.ndarray, list], e: Union[np.ndarray, list]):
		self.m1 = self.m10.copy()
		self.m2 = self.m20.copy()
		self.m3 = self.m30.copy()
		self.n1 = self.n10.copy()
		self.n2 = self.n20.copy()
		self.n3 = self.n30.copy()

		obs_e = e - self.z1
		self.dz1 = self.z2 + self.m1 * self.sig(obs_e, self.a1) + self.n1 * self.sig(obs_e, self.b1)
		self.dz2 = syst_dynamic + self.z3 + self.m2 * self.sig(obs_e, self.a2) + self.n2 * self.sig(obs_e, self.b2)
		self.dz3 = self.m3 * self.sig(obs_e, self.a3) + self.n3 * self.sig(obs_e, self.b3)
		self.z1 = self.z1 + self.dz1 * self.dt
		self.z2 = self.z2 + self.dz2 * self.dt
		self.z3 = self.z3 + self.dz3 * self.dt

		return self.z3, self.dz3
