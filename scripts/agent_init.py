import rospy
from mavros_msgs.msg import State, AttitudeTarget, ParamValue
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, ParamSet, ParamSetRequest

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from sensor_msgs.msg import BatteryState
from sensor_msgs.msg import Imu

import tf
from tf.transformations import quaternion_matrix

import numpy as np
import pandas as pd
from typing import Union

from smc_ctrl.ref_cmd import *
from smc_ctrl.utils import *
from smc_ctrl.FNTSMC import fntsmc_param, fntsmc_pos
from smc_ctrl.uav_ros import UAV_ROS
from smc_ctrl.observer import neso
from smc_ctrl.observer import robust_differentiator_3rd as rd3


class fntsmc_ppo_ros:
	def __init__(self, dt: float=0.01):
		self.current_state = State()  # monitor uav status
		self.pose = PoseStamped()  # publish offboard [x_d y_d z_d] cmd
		self.uav_odom = Odometry()  # subscribe uav state x y z vx vy vz phi theta psi p q r
		self.ctrl_cmd = AttitudeTarget()  # publish offboard expected [phi_d theta_d psi_d throttle] cmd
		self.voltage = 11.4  # subscribe voltage from the battery
		self.global_flag = 0  # UAV working mode monitoring
		self.DT = dt
		self.use_gazebo = True
		self.pkg_name = 'pid_ros'

		''' 选择不同的控制器 '''
		# self.controller = 'FNTSMC'
		self.controller = 'DUAL-PID'
		# self.controller = 'RL'
		# self.controller = 'PX4-PID'
		# self.controller = 'MPC'
		''' 选择不同的控制器 '''

		'''选择不同观测器'''
		self.observer = 'rd3'
		# self.observer = 'neso'
		# self.observer = 'none'
		self.obs = None
		'''选择不同观测器'''


		rospy.init_node("offb_node_py")  # 初始化一个节点
		self.state_sub = rospy.Subscriber("mavros/state", State, callback=self.state_cb)
		self.uav_vel_sub = rospy.Subscriber("mavros/local_position/odom", Odometry, callback=self.uav_odom_cb)
		self.uav_battery_sub = rospy.Subscriber("mavros/battery", BatteryState, callback=self.uav_battery_cb)
		self.uav_rate_sub = rospy.Subscriber("mavros/imu/data", Imu, callback=self.uav_rate_cb)
		'''topic subscriberate'''

		self.local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
		self.uav_att_throttle_pub = rospy.Publisher("mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)
		'''Publish 位置指令给 UAV'''

		'''arming service'''
		rospy.wait_for_service("/mavros/cmd/arming")  # 等待解锁电机的 service 建立
		self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

		'''working mode service'''
		rospy.wait_for_service("/mavros/set_mode")  # 等待设置 UAV 工作模式的 service 建立
		self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

		self.rate = rospy.Rate(1 / self.DT)
		self.offb_set_mode = None
		self.arm_cmd = None

		self.t0 = rospy.Time.now().to_sec()

	def initialization(self):
		print('initialization')
		while (not rospy.is_shutdown()) and (not self.current_state.connected):
			self.rate.sleep()
		self.pose.pose.position.x = self.uav_odom.pose.pose.position.x
		self.pose.pose.position.y = self.uav_odom.pose.pose.position.y
		self.pose.pose.position.z = self.uav_odom.pose.pose.position.z

		for i in range(100):
			if rospy.is_shutdown():
				break
			self.local_pos_pub.publish(self.pose)
			self.rate.sleep()

		self.offb_set_mode = SetModeRequest()
		self.offb_set_mode.custom_mode = 'OFFBOARD'  # 先设置工作模式为 offboard

		self.arm_cmd = CommandBoolRequest()
		self.arm_cmd.value = True  # 通过指令将电机解锁

		while (self.current_state.mode != "OFFBOARD") and (not rospy.is_shutdown()):  # 等待
			if self.set_mode_client.call(self.offb_set_mode).mode_sent:
				print('Switching to OFFBOARD mode is available...waiting for 1 seconds')
				break
			self.local_pos_pub.publish(self.pose)
			self.rate.sleep()

		self.t0 = rospy.Time.now().to_sec()

		while rospy.Time.now().to_sec() - self.t0 < 1.0:
			self.local_pos_pub.publish(self.pose)
			self.rate.sleep()

		while (not self.current_state.armed) and (not rospy.is_shutdown()):
			if self.arming_client.call(self.arm_cmd).success:
				print('UAV is armed now...waiting for 1 seconds')
				break
			self.local_pos_pub.publish(self.pose)
			self.rate.sleep()

		self.t0 = rospy.Time.now().to_sec()

		while rospy.Time.now().to_sec() - self.t0 < 1.0:  # OK
			self.local_pos_pub.publish(self.pose)
			self.rate.sleep()

		print('Start......')
		print('Approaching...')
		self.global_flag = 1

		self.t0 = rospy.Time.now().to_sec()

	def state_cb(self, msg: State):
		self.current_state = msg

	def uav_odom_cb(self, msg: Odometry):
		self.uav_odom.pose.pose.position.x = msg.pose.pose.position.x
		self.uav_odom.pose.pose.position.y = msg.pose.pose.position.y
		self.uav_odom.pose.pose.position.z = msg.pose.pose.position.z
		self.uav_odom.pose.pose.orientation.x = msg.pose.pose.orientation.x
		self.uav_odom.pose.pose.orientation.y = msg.pose.pose.orientation.y
		self.uav_odom.pose.pose.orientation.z = msg.pose.pose.orientation.z
		self.uav_odom.pose.pose.orientation.w = msg.pose.pose.orientation.w

		self.uav_odom.twist.twist.linear.x = msg.twist.twist.linear.x
		self.uav_odom.twist.twist.linear.y = msg.twist.twist.linear.y
		self.uav_odom.twist.twist.linear.z = msg.twist.twist.linear.z
		self.uav_odom.twist.twist.angular.x = msg.twist.twist.angular.x
		self.uav_odom.twist.twist.angular.y = msg.twist.twist.angular.y
		self.uav_odom.twist.twist.angular.z = msg.twist.twist.angular.z

  

	def uav_battery_cb(self, msg: BatteryState):
		self.voltage = msg.voltage
  
	def uav_rate_cb(self, data: Imu):
    	# 获取角速度
		self.angular_velocity = data.angular_velocity
		self.roll_rate = self.angular_velocity.x
		self.pitch_rate = self.angular_velocity.y
		self.yaw_rate = self.angular_velocity.z

	def uav_odom_2_uav_state(self, odom: Odometry) -> np.ndarray:
		_orientation = odom.pose.pose.orientation
		_w = _orientation.w
		_x = _orientation.x
		_y = _orientation.y
		_z = _orientation.z
		rpy = tf.transformations.euler_from_quaternion([_x, _y, _z, _w])
		_uav_state = np.array([
			odom.pose.pose.position.x,  # x
			odom.pose.pose.position.y,  # y
			odom.pose.pose.position.z,  # z
			odom.twist.twist.linear.x,  # vx
			odom.twist.twist.linear.y,  # vy
			odom.twist.twist.linear.z,  # vz
			rpy[0],  # phi
			rpy[1],  # theta
			rpy[2],  # psi
			odom.twist.twist.angular.x,  # p
			odom.twist.twist.angular.y,  # q
			odom.twist.twist.angular.z  # r
		])
		return _uav_state

	def thrust_2_throttle(self, thrust: float):
		"""

		"""
		'''线性模型'''
		if self.use_gazebo:
			k = 0.46 / 0.72 / 9.8
		else:
			k = 0.31 / 0.727 / 9.8
		_throttle = max(min(k * thrust, 2.0), 0.01)
		'''线性模型'''
		return _throttle

	def euler_2_quaternion(self, phi, theta, psi):
		w = C(phi / 2) * C(theta / 2) * C(psi / 2) + S(phi / 2) * S(theta / 2) * S(psi / 2)
		x = S(phi / 2) * C(theta / 2) * C(psi / 2) - C(phi / 2) * S(theta / 2) * S(psi / 2)
		y = C(phi / 2) * S(theta / 2) * C(psi / 2) + S(phi / 2) * C(theta / 2) * S(psi / 2)
		z = C(phi / 2) * C(theta / 2) * S(psi / 2) - S(phi / 2) * S(theta / 2) * C(psi / 2)
		return [x, y, z, w]

	def get_normalizer_from_file(self, dim, path, file):
		norm = Normalization(dim)
		data = pd.read_csv(path + file, header=0).to_numpy()
		norm.running_ms.n = data[0, 0]
		norm.running_ms.mean = data[:, 1]
		norm.running_ms.std = data[:, 2]
		norm.running_ms.S = data[:, 3]
		norm.running_ms.n = data[0, 4]
		norm.running_ms.mean = data[:, 5]
		norm.running_ms.std = data[:, 6]
		norm.running_ms.S = data[:, 7]
		return norm

	def position_ctrl_with_PX4(self, _pose: Union[list, np.ndarray], _q: Union[list, np.ndarray] = None):
		"""
		_pose: x y z
		_q: x y z w
		"""
		self.pose.pose.position.x = _pose[0]
		self.pose.pose.position.y = _pose[1]
		self.pose.pose.position.z = _pose[2]
		if _q is not None:
			self.pose.pose.orientation.x = _q[0]
			self.pose.pose.orientation.y = _q[1]
			self.pose.pose.orientation.z = _q[2]
			self.pose.pose.orientation.w = _q[3]
		self.local_pos_pub.publish(self.pose)

	def approaching(self, ref_a: np.ndarray, ref_p: np.ndarray, ref_ba: np.ndarray, ref_bp: np.ndarray):
		# ref_a = np.array([0., 0., 0., 0.])
		_ref, _, _, _ = ref_uav(0., ref_a, ref_p, ref_ba, ref_bp)

		cmd_q = tf.transformations.quaternion_from_euler(0., 0., _ref[3])
		uav_state = self.uav_odom_2_uav_state(self.uav_odom)
		self.position_ctrl_with_PX4(_pose=_ref[0: 3], _q=cmd_q)

		# self.path_msg.poses.append(self.pose)
		# self.pub_path.publish(self.path_msg)
		
		if (np.linalg.norm(_ref[0: 3] - uav_state[0: 3]) < 0.3) and \
				(np.linalg.norm(_ref[3] - uav_state[8]) < deg2rad(5)) and \
				(np.linalg.norm(uav_state[3: 6]) < 0.3):
			return True
		return False

	def set_observer(self, uav_ros: UAV_ROS):
		if self.observer == 'neso':
			self.obs = neso(l1=np.array([0.1, 0.1, 0.2]),
							l2=np.array([0.1, 0.1, 0.2]),
							l3=np.array([0.08, 0.08, 0.08]),
							r=np.array([0.25, 0.25, 0.25]),  # r 越小，增益越小 奥利给兄弟们干了
							k1=np.array([0.7, 0.7, 0.7]),
							k2=np.array([0.01, 0.01, 0.01]),
							dim=3,
							dt=self.DT)
			syst_dynamic_out = -uav_ros.kt / uav_ros.m * uav_ros.dot_eta() + uav_ros.A()
			self.obs.set_init(x0=uav_ros.eta(), dx0=uav_ros.dot_eta(), syst_dynamic=syst_dynamic_out)
		elif self.observer == 'rd3':
			self.obs = rd3(use_freq=True,
						   omega=[[0.9, 0.9, 0.9],
								  [0.9, 0.9, 0.9],
								  [0.9, 0.9, 0.9]],
						   dim=3, dt=self.DT)
			syst_dynamic_out = -uav_ros.kt / uav_ros.m * uav_ros.dot_eta() + uav_ros.A()
			self.obs.set_init(e0=uav_ros.eta(), de0=uav_ros.dot_eta(), syst_dynamic=syst_dynamic_out)
		else:
			self.obs = None

	def observe(self, uav_ros: UAV_ROS):
		if self.observer == 'neso':
			syst_dynamic = -uav_ros.kt / uav_ros.m * uav_ros.dot_eta() + uav_ros.A()
			res, _ = self.obs.observe(x=uav_ros.eta(), syst_dynamic=syst_dynamic)
		elif self.observer == 'rd3':
			syst_dynamic = -uav_ros.kt / uav_ros.m * uav_ros.dot_eta() + uav_ros.A()
			res, _ = self.obs.observe(e=uav_ros.eta(), syst_dynamic=syst_dynamic)
		else:
			res = np.zeros(3)
		return res
	def publish_control_cmd(self, phi_d: float, theta_d: float, psi_d: float, uf: float):
		self.ctrl_cmd.header.stamp = rospy.Time.now()
		self.ctrl_cmd.type_mask = AttitudeTarget.IGNORE_ROLL_RATE + AttitudeTarget.IGNORE_PITCH_RATE + AttitudeTarget.IGNORE_YAW_RATE
		cmd_q = tf.transformations.quaternion_from_euler(phi_d, theta_d, psi_d, axes='sxyz')
		self.ctrl_cmd.orientation.x = cmd_q[0]
		self.ctrl_cmd.orientation.y = cmd_q[1]
		self.ctrl_cmd.orientation.z = cmd_q[2]
		self.ctrl_cmd.orientation.w = cmd_q[3]
		self.ctrl_cmd.thrust = self.thrust_2_throttle(uf)
		self.uav_att_throttle_pub.publish(self.ctrl_cmd)

	def publish_rate_control_cmd(self, phi_v_d: float, theta_v_d: float, psi_v_d: float, uf: float):
		self.ctrl_cmd.header.stamp = rospy.Time.now()
		self.ctrl_cmd.type_mask = AttitudeTarget.IGNORE_ATTITUDE
		
		self.ctrl_cmd.body_rate.x = phi_v_d
		self.ctrl_cmd.body_rate.y = theta_v_d
		self.ctrl_cmd.body_rate.z = psi_v_d
  
		self.ctrl_cmd.thrust = self.thrust_2_throttle(uf)
		self.uav_att_throttle_pub.publish(self.ctrl_cmd)

	def ros_sleep_sec(self, t: float):

		self.pose.pose.position.x = self.uav_odom.pose.pose.position.x
		self.pose.pose.position.y = self.uav_odom.pose.pose.position.y
		self.pose.pose.position.z = self.uav_odom.pose.pose.position.z

		for i in range(int(t / self.DT)):
			if rospy.is_shutdown():
				break
			self.local_pos_pub.publish(self.pose)
			self.rate.sleep()

def set_pid_param(param_id, value):
	rospy.wait_for_service('/mavros/param/set')
	try:
		param_set_service = rospy.ServiceProxy('/mavros/param/set', ParamSet)
		param_value = ParamValue()
		param_value.real = value
		param_set_request = ParamSetRequest()
		param_set_request.param_id = param_id
		param_set_request.value = param_value
		response = param_set_service(param_set_request)
		if response.success:
			rospy.loginfo(f"Successfully set {param_id} to {value}")
		else:
			rospy.logerr(f"Failed to set {param_id}")
	except rospy.ServiceException as e:
		rospy.logerr(f"Service call failed: {e}")
  
def set_rate_pid_para(r_p=0.35,
                      p_p=0.35,
                      y_p=0.2,
                      r_i=0.2,
                      p_i=0.2,
                      y_i=0.2):
	    # Example: Set roll rate P gain
    set_pid_param('MC_ROLLRATE_P', r_p)
    set_pid_param('MC_ROLLRATE_I', r_i)
    
    # Example: Set pitch rate P gain
    set_pid_param('MC_PITCHRATE_P', p_p)
    set_pid_param('MC_PITCHRATE_I', p_i)

    # Example: Set yaw rate P gain
    set_pid_param('MC_YAWRATE_P', y_p)
    set_pid_param('MC_YAWRATE_I', y_i)
    
    