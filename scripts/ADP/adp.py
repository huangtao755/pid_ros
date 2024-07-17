# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
# import torch
# import torch as t
# from torch.autograd import Variable



class PIDSMADP(object):
    def __init__(self,
                 sm_k1=100,
                 sm_k2=1000,
                 Q=np.diag([10, 1, 1]),
                 R=np.diag([0.01, 0.01])):
        self.kp = 0.
        self.kd = 0.

        self.u_kp = 0
        self.u_kd = 0
        self.u = np.array([self.u_kp, self.u_kd])

        self.r = np.diag([1, 1])

        self.w = 0.01*np.ones(6)
        self.phi = np.zeros(6)
        self.d_phi = np.zeros((3, 6))
        
        self.sm_k1 = sm_k1
        self.sm_k2 = sm_k2
        
        self.Q = Q
        self.R = R
        
        self.s = 0
        self.r = 0
        
    def update_sm(self, state, ref_state):
        self.s = self.sm_k1 * (ref_state[0] - state[0]) + self.sm_k2 * (ref_state[1] - state[1])
        
    def update_reward(self, state, v):
        self.r = state.dot(self.Q).dot(state.T) + v.dot(self.R).dot(v.T)
        
    def get_phi(self, state):
        s = state[0]
        kp = state[1]
        kd = state[2]
        self.phi = np.array([s**2, kp**2, kd**2, s*kp, s*kd, kp*kd])

    def get_d_phi(self, state):
        s = state[0]
        kp = state[1]
        kd = state[2]
        self.d_phi = np.array([[2*s, 0, 0, kp, kd, 0],
                               [0, 2*kp, 0, s, 0, kd],
                               [0, 0, 2*kd, 0, s, kp]])

    def u_star(self, state, dt=0.01):
        self.get_d_phi(state)

        self.u_kp = -1/2 / self.r[0, 0] * self.d_phi[1, :].dot(self.w)
        self.u_kd = -1/2 / self.r[1, 1] * self.d_phi[2, :].dot(self.w)

        print(self.u_kp, 'u_kp')
        print(self.u_kp > 0)
        print(self.kp, 'kp_old')
        self.kp = self.kp + self.u_kp*dt

        print(self.kp, 'kpp')

        self.kd = self.kd + self.u_kd * dt
        self.u = np.array([self.u_kp, self.u_kd])
        return self.u_kp, self.u_kd

    def reset(self):
        self.kp = 0
        self.kd = 0

    def trainI(self, state, state_, r, v, dt=0.01, lr=0.001):
        r = np.array(r)
        v = np.array(v)
        r = r + v.dot(self.r).dot(v.T)

        self.get_phi(state)
        phi = self.phi
        self.get_phi(state_)
        phi_ = self.phi
        delta_phi = phi_ - phi
        print(delta_phi, 'delta_phi')
        e_h = r * dt + self.w @ delta_phi + np.random.randn(1)*0.00

        m = delta_phi.T@delta_phi + 1
        print(m, 'm')
        self.w = self.w - lr * delta_phi/m**2*e_h
        print(self.w, 'w')


class SMADP(object):
    def __init__(self, 
                 sm_k1=100,
                 sm_k2=1000):
        self.u = 0.

        self.v = 0.

        self.r = 0.1

        self.w = 0.0001*np.ones(3)
        self.phi = np.zeros(3)
        self.d_phi = np.zeros((2, 3))
        
        self.sm_k1 = 100
        self.sm_k2 = 1000
        
        self.s = 0

    def get_phi(self, state):
        s = state[0]
        u = state[1]
        self.phi = np.array([s**2, u**2, s*u])

    def get_d_phi(self, state):
        s = state[0]
        u = state[1]
        self.d_phi = np.array([[2*s, 0, u],
                               [0, 2*u, s]])

    def u_star(self, state, dt=0.01):
        self.get_d_phi(state)

        self.v = -1/2 / self.r * self.d_phi[1, :].dot(self.w)

        print(self.u_kp, 'u_kp')
        print(self.u_kp > 0)
        print(self.kp, 'kp_old')
        self.kp = self.kp + self.u_kp*dt

        print(self.kp, 'kpp')

        self.kd = self.kd + self.u_kd * dt
        self.u = np.array([self.u_kp, self.u_kd])
        return self.u_kp, self.u_kd

    def reset(self):
        self.kp = 0
        self.kd = 0

    def trainI(self, state, state_, r, v, dt=0.01, lr=0.01):
        r = np.array(r)
        v = np.array(v)
        r = r + v.dot(self.r).dot(v.T)

        self.get_phi(state)
        phi = self.phi
        self.get_phi(state_)
        phi_ = self.phi
        delta_phi = phi_ - phi
        print(delta_phi, 'delta_phi')
        e_h = r * dt + self.w @ delta_phi + np.random.randn(1)*0.001

        m = delta_phi.T@delta_phi + 1
        print(m, 'm')
        self.w = self.w - lr * delta_phi/m**2*e_h
        print(self.w, 'w')

    def get_sm(self, state):
        return self.sm_k1 * state[0] + self.sm_k2 * state[1]    

class PSMADP(object):
    def __init__(self):
        self.kp = 0.

        self.u_kp = 0
        self.u = np.array([self.u_kp])

        self.r = np.diag([1])

        self.w = 0.01*np.ones(3)
        self.phi = np.zeros(3)
        self.d_phi = np.zeros((3, 3))

    def get_phi(self, state):
        s = state[0]
        kp = state[1]
        self.phi = np.array([s**2, kp**2, s*kp])

    def get_d_phi(self, state):
        s = state[0]
        kp = state[1]
        self.d_phi = np.array([[2*s, 0, kp],
                               [0, 2*kp, s]])

    def u_star(self, state, dt=0.01):
        self.get_d_phi(state)
        print(self.r)
        print(self.d_phi[1, :])
        print(self.w)
        self.u_kp = -1/2 / self.r[0] * self.d_phi[1, :].dot(self.w)

        print(self.u_kp, 'u_kp')
        print(self.u_kp > 0)
        print(self.kp, 'kp_old')
        self.kp = self.kp + self.u_kp*dt

        print(self.kp, 'kpp')

        self.u = np.array([self.u_kp])
        return self.u_kp

    def reset(self):
        self.kp = 0

    def trainI(self, state, state_, r, v, dt=0.01, lr=0.001):
        r = np.array(r)
        v = np.array(v)
        r = r + v.dot(self.r).dot(v.T)

        self.get_phi(state)
        phi = self.phi
        self.get_phi(state_)
        phi_ = self.phi
        delta_phi = phi_ - phi
        print(delta_phi, 'delta_phi')
        e_h = r * dt + self.w @ delta_phi + np.random.randn(1)*0.00

        m = delta_phi.T@delta_phi + 1
        print(m, 'm')
        self.w = self.w - lr * delta_phi/m**2*e_h
        print(self.w, 'w')