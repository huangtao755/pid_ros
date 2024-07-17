#! /usr/bin/python3
import os


import rospy

from smc_ctrl.uav_ros import UAV_ROS
from smc_ctrl.FNTSMC import fntsmc_param, fntsmc_pos
from smc_ctrl.collector import data_collector

from smc_ctrl.ref_cmd import *
from smc_ctrl.utils import *
from scripts.agent_init import fntsmc_ppo_ros

from dual_pid_ctrl.dual_PID import PidControl
print('No errors while importing python modules...')


# '''Parameter list of the position controller 跟踪'''
# DT = 0.01
# pos_ctrl_param = fntsmc_param()
# pos_ctrl_param.k1 = np.array([1.2, 0.8, 4.0])		# optimal: 1.2, 0.8, 4.0
# pos_ctrl_param.k2 = np.array([1.4, 1.8, 0.9])		# optimal: .6, 1.0, 0.5
# pos_ctrl_param.alpha = np.array([1.2, 1.5, 2.5])	# optimal: 1.2, 1.5, 2.5
# pos_ctrl_param.beta = np.array([0.6, 0.9, 0.75])	# optimal: 0.6, 0.6, 0.75
# pos_ctrl_param.gamma = np.array([0.2, 0.2, 0.2])	# optimal: 0.2, 0.2, 0.2
# pos_ctrl_param.lmd = np.array([1.0, 1.0, 1.0])		# optimal: 2.0, 2.0, 2.0
# pos_ctrl_param.vel_c = np.array([0.18, 0.2, 0.3])
# # pos_ctrl_param.vel_c = np.array([0.3, 0.4, 0.0])  # 0.05, 0.05, -0.005	昨晚gazebo
# pos_ctrl_param.acc_c = np.array([0.5, 0.5, 0.])
# pos_ctrl_param.dim = 3
# pos_ctrl_param.dt = DT
# pos_ctrl_param.ctrl0 = np.array([0., 0., 0.])
# pos_ctrl_param.saturation = np.array([np.inf, np.inf, np.inf])
# '''Parameter list of the position controller 跟踪'''

'''Parameter list of the position controller 定点'''
DT = 0.01
pos_ctrl_param = fntsmc_param()
pos_ctrl_param.k1 = np.array([1.2, 0.8, 4.0])		# optimal: 1.2, 0.8, 4.0
pos_ctrl_param.k2 = np.array([0.6, 1.0, 0.5])		# optimal: .6, 1.0, 0.5
pos_ctrl_param.alpha = np.array([1.2, 1.5, 2.5])	# optimal: 1.2, 1.5, 2.5
pos_ctrl_param.beta = np.array([0.6, 0.9, 0.75])	# optimal: 0.6, 0.6, 0.75
pos_ctrl_param.gamma = np.array([0.2, 0.2, 0.2])	# optimal: 0.2, 0.2, 0.2
pos_ctrl_param.lmd = np.array([1.0, 1.0, 1.0])		# optimal: 2.0, 2.0, 2.0
pos_ctrl_param.vel_c = np.array([0., 0., 0.3])
pos_ctrl_param.acc_c = np.array([0., 0., 0.])
pos_ctrl_param.dim = 3
pos_ctrl_param.dt = DT
pos_ctrl_param.ctrl0 = np.array([0., 0., 0.])
pos_ctrl_param.saturation = np.array([np.inf, np.inf, np.inf])
'''Parameter list of the position controller 定点'''


if __name__ == "__main__":
	agent = fntsmc_ppo_ros(dt=0.01)
	agent.initialization()

	uav_ros = None
	fntsmc_ctrl = None
	data_record = None


	ref_period = np.array([8, 8, 10, 15])  # xd yd zd psid 周期
	ref_bias_a = np.array([2.0, 2.0, 0.5, deg2rad(0)])  # xd yd zd psid 幅值偏移
	ref_bias_phase = np.array([np.pi / 2, 0, 0, 0])  # xd yd zd psid 相位偏移

	while not rospy.is_shutdown():
		t = rospy.Time.now().to_sec()
		if agent.global_flag == 1:  # approaching
			if agent.approaching(np.zeros(4), ref_period, ref_bias_a, ref_bias_phase):
				agent.global_flag = 2
		elif agent.global_flag == 2:	# preparing
			agent.ros_sleep_sec(2)		# wait for 2 seconds
			print('Final preparing for control.')
			uav_ros = UAV_ROS(m=0.722, g=9.8, kt=1e-3, dt=agent.DT, time_max=30)  # 0.722
			fntsmc_ctrl = fntsmc_pos(pos_ctrl_param)
			agent.set_observer(uav_ros)
			data_record = data_collector(N=round(uav_ros.time_max / DT))
			agent.t0 = rospy.Time.now().to_sec()
			uav_ros.set_state(agent.uav_odom_2_uav_state(agent.uav_odom))
			agent.global_flag = 3
		elif agent.global_flag == 3:  # control
			t_now = round(t - agent.t0, 4)
			if uav_ros.n % 100 == 0:
				print('time: ', t_now)

			'''1. generate reference command and uncertainty'''
			# rax = max(min(0.4 * t_now, 1.3), 0.)  # 1.5
			# ray = max(min(0.4 * t_now, 1.3), 0.)  # 1.5
			# raz = max(min(0.075 * t_now, 0.3), 0.)  # 0.3
			# rapsi = max(min(deg2rad(10) / 5 * t_now, deg2rad(10)), 0.0)  # pi / 3
			rax = 0.
			ray = 0.
			raz = 0.
			rapsi = deg2rad(0)
			ref_amplitude = np.array([rax, ray, raz, rapsi])
			ref_period = np.array([8, 8, 10, 15])  # xd yd zd psid 周期
			ref_bias_a = np.array([0, 0, 1.1, deg2rad(0)])  # xd yd zd psid 幅值偏移
			ref_bias_phase = np.array([np.pi / 2, 0, 0, 0])  # xd yd zd psid 相位偏移
			ref, dot_ref, dot2_ref, dot3_ref = ref_uav(t_now,
													   ref_amplitude,
													   ref_period,
													   ref_bias_a,
													   ref_bias_phase)

			'''2. generate outer-loop reference signal 'eta_d' and its 1st, 2nd, and 3rd-order derivatives'''
			eta_d = ref[0: 3]
			dot_eta_d = dot_ref[0: 3]
			dot2_eta_d = dot2_ref[0: 3]
			e = uav_ros.eta() - eta_d
			de = uav_ros.dot_eta() - dot_eta_d
			psi_d = ref[3]

			observe = agent.observe(uav_ros)

			'''3. Update the parameters of FNTSMC if RL is used'''
			if agent.controller == 'PX4-PID':
				agent.position_ctrl_with_PX4(ref[0: 3])
				phi_d, theta_d, uf = 0., 0., 0.		# 缺省，无意义
			else:
				'''3.1 generate phi_d, theta_d, throttle'''
				fntsmc_ctrl.control_update(uav_ros.kt, uav_ros.m, uav_ros.uav_vel(), e, de, dot_eta_d, dot2_eta_d, obs=observe)

				phi_d, theta_d, uf = uo_2_ref_angle_throttle(fntsmc_ctrl.control,
															 uav_ros.uav_att(),
															 psi_d,
															 uav_ros.m,
															 uav_ros.g,
															 limit=[np.pi / 6, np.pi / 6],
															 att_limitation=True)

				'''4. publish control cmd'''
				agent.publish_control_cmd(phi_d, theta_d, psi_d, uf)

			'''5. get new uav states from Gazebo'''
			uav_ros.rk44(action=[phi_d, theta_d, uf], uav_state=agent.uav_odom_2_uav_state(agent.uav_odom))

			'''6. data storage'''
			data_block = {'time': uav_ros.time,  # simulation time
						  'throttle': uf,
						  'thrust': agent.ctrl_cmd.thrust,
						  'ref_angle': np.array([phi_d, theta_d, psi_d]),
						  'ref_pos': ref[0: 3],
						  'ref_vel': dot_ref[0: 3],
						  'd_out_obs': observe,
						  'state': uav_ros.uav_state_call_back(),
						  'dot_angle': uav_ros.uav_dot_att()}
			data_record.record(data_block)

			if data_record.index == data_record.N:
				print('Data collection finish. Switching offboard position...')
				data_record.package2file(path=os.getcwd() + '/src/' + agent.pkg_name + '/scripts/datasave/')
				agent.global_flag = 4
		elif agent.global_flag == 4:  # finish, back to offboard position
			agent.position_ctrl_with_PX4([0., 0., 0.5])
		else:
			agent.position_ctrl_with_PX4([0., 0., 0.5])
			print('WORKING MODE ERROR...')
		agent.rate.sleep()
