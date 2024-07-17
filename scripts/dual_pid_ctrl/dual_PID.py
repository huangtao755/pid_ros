import numpy as np

class PidControl(object):
    def __init__(self,
                 dt: float=0.01,
                 kp_pos: np.ndarray=np.zeros(3),
                 ki_pos: np.ndarray=np.zeros(3),
                 kd_pos: np.ndarray=np.zeros(3),
                 kp_vel: np.ndarray=np.zeros(3),
                 ki_vel: np.ndarray=np.zeros(3),
                 kd_vel: np.ndarray=np.zeros(3),
                 kp_att: np.ndarray=np.zeros(3),
                 ki_att: np.ndarray=np.zeros(3),
                 kd_att: np.ndarray=np.zeros(3),
                 p_v: np.ndarray=np.ones(3),
                 p_a: np.ndarray=np.ones(3),
                 p_r: np.ndarray=np.ones(3)):

        " init model "
        self.dt = dt
        " init model "
        
        " init control para "
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos

        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        self.kd_vel = kd_vel

        self.kp_att = kp_att
        self.ki_att = ki_att
        self.kd_att = kd_att

        self.p_v = p_v
        self.p_a = p_a
        self.p_r = p_r
        " init control para "
        
        " simulation state "
        self.err_p_pos = np.zeros(3)
        self.err_i_pos = np.zeros(3)
        self.err_d_pos = np.zeros(3)

        self.err_p_vel = np.zeros(3)
        self.err_i_vel = np.zeros(3)
        self.err_d_vel = np.zeros(3)

        self.err_p_att = np.zeros(3)
        self.err_i_att = np.zeros(3)
        self.err_d_att = np.zeros(3)

        self.err_control = np.zeros(3)
        self.control = np.zeros(3)
        
        self.att_control = np.zeros(3)
        " simulation state "
    
    def para(self):
        print('Para for fix PID',
            'kp_pos:', self.kp_pos,
            'ki_pos:', self.ki_pos,
            'kd_pos:', self.kd_pos,
            'kp_vel:', self.kp_vel,
            'ki_vel:', self.ki_vel,
            'kd_vel:', self.kd_vel,
            'alpha_pos:', self.p_v,
            'alpha_vel:', self.p_a)
        
    def update_att(self, 
                   uf: float,
                   att: np.ndarray,
                   pqr: np.ndarray,
                   ref_att: np.ndarray = np.zeros(3),
                   ref_att_old: np.ndarray = np.zeros(3),
                   ref_att_v: np.ndarray = np.zeros(3)):
        " Attitude loop "
        # print(att, 'att')
    
        ref_att_v = (ref_att - ref_att_old) / self.dt
        
        self.err_p_att = (ref_att - att)# .clip(np.array([-1.57/3, -1.57/3, -1.57/3]),np.array([1.57/3, 1.57/3, 1.57/3]))
        
        for i in range(3):
            self.err_i_att[i] += abs(self.err_p_att[i].clip(-0.1, 0.1))**self.p_r[i]*np.tanh(20*self.err_p_att[i])*self.dt
        
        self.err_d_att = ref_att_v - pqr

        att_v = np.zeros(3)
        for i in range(3):
            att_v[i] = self.kp_att[i] * abs(self.err_p_att[i])**(self.p_r[i])*np.tanh(20*self.err_p_att[i]) \
                + self.ki_att[i] * self.err_i_att[i] \
                + self.kd_att[i] * abs(self.err_d_att[i])**(2*self.p_r[i]/(1+self.p_r[i]))*np.tanh(10*self.err_p_att[i])
            
        att_v = att_v.clip(np.array([-3, -3, -3]), np.array([3, 3, 3])) + 0.*ref_att_v
        self.att_control = att_v 
        return self.att_control[0], self.att_control[1], self.att_control[2]
        " Attitude loop "        
    
    def update_dual(self,
               state: np.ndarray,
               ref_state: np.ndarray=np.zeros(9)) -> np.ndarray:
        " position loop "
        # self.err_p_pos = 5 * np.tanh(0.2*(ref_state[0: 3] - state[0: 3]))
        self.err_p_pos = (ref_state[0: 3] - state[0: 3]).clip(np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5]))
        
        for i in range(3):
            self.err_i_pos[i] += abs(self.err_p_pos[i].clip(-0.1, 0.1))**self.p_v[i]*np.tanh(100*self.err_p_pos[i])*self.dt

        # self.err_d_pos = 10 * np.tanh(0.1*(ref_state[3: 6] - state[3: 6])/self.dt)
        self.err_d_pos = (ref_state[3: 6] - state[3: 6])  

        pos_vel = np.zeros(3)
        for i in range(3):
            pos_vel[i] = self.kp_pos[i] * abs(self.err_p_pos[i])**self.p_v[i]*np.tanh(50*self.err_p_pos[i]) \
                  + self.ki_pos[i] * self.err_i_pos[i] \
                  + self.kd_pos[i] * abs(self.err_d_pos[i])**(2*self.p_v[i]/(1+self.p_v[i]))*np.tanh(100*self.err_d_pos[i])
        # print(pos_vel, 'pos_vel')
            
        pos_vel = pos_vel + ref_state[3: 6]
        " position loop "

        " Velocity loop "
        self.err_p_vel = (pos_vel - state[3: 6])
        self.err_i_vel += self.err_p_vel * self.dt
        self.err_d_vel = (pos_vel - state[3: 6])/self.dt

        vel_a = np.zeros(3)
        for i in range(3):
            vel_a[i] = self.kp_vel[i] * abs(self.err_p_vel[i])**(self.p_a[i])*np.tanh(50*self.err_p_vel[i]) \
                + self.ki_vel[i] * self.err_i_vel[i] \
                + self.kd_vel[i] * abs(self.err_d_vel[i])**(2*self.p_a[i]/(1+self.p_a[i]))*np.tanh(100*self.err_p_vel[i])
            
        vel_a = vel_a.clip(np.array([-10, -10, -10]), np.array([10, 10, 10]))
        vel_a += ref_state[6: 9]
        
        self.control = 0.2*self.control + 0.8*vel_a 
        " Velocity loop "

    def update(self,
            state: np.ndarray,
            ref_state: np.ndarray=np.zeros(9)) -> np.ndarray:
        self.err_p_pos = (ref_state[0: 3] - state[0: 3]).clip(np.array([-0.5, -0.5, -0.4]), np.array([0.5, 0.5, 0.4]))
        
        for i in range(3):
            self.err_i_pos[i] = self.err_i_pos[i]*0.99 + abs(self.err_p_pos[i].clip(-0.01, 0.01))**self.p_v[i]*np.sign(40*self.err_p_pos[i])*self.dt
        # self.err_i_pos += self.err_p_pos * self.dt 

        # self.err_d_pos = 10 * np.tanh(0.1*(ref_state[3: 6] - state[3: 6])/self.dt)
        self.err_d_pos = (ref_state[3: 6] - state[3: 6])  

        a_vel = np.zeros(3)
        for i in range(3):
            a_vel[i] = self.kp_pos[i] * abs(self.err_p_pos[i])**self.p_v[i]*np.tanh(100*self.err_p_pos[i]) \
                  + self.ki_pos[i] * self.err_i_pos[i] \
                  + self.kd_pos[i] * abs(self.err_d_pos[i])**(2*self.p_v[i]/(1+self.p_v[i]))*np.tanh(40*self.err_d_pos[i])
        # print(pos_vel, 'pos_vel')
            
        a_vel = a_vel.clip(np.array([-10, -10, -10]), np.array([10, 10, 10]))  
        a_vel +=  ref_state[6: 9]
        self.err_control = 0.15*self.err_control + 0.85 * a_vel
        self.control = self.err_control # + ref_state[6: 9]
        
    def reset(self):
        " simulation state "
        self.err_p_pos = np.zeros(3)
        self.err_i_pos = np.zeros(3)
        self.err_d_pos = np.zeros(3)

        self.err_p_vel = np.zeros(3)
        self.err_i_vel = np.zeros(3)
        self.err_d_vel = np.zeros(3)
        " simulation state "


