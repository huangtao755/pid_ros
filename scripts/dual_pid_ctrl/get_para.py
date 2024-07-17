#!/usr/bin/env python

import rospy
from mavros_msgs.srv import ParamGet

def get_param(param_id):
    rospy.wait_for_service('/mavros/param/get')
    try:
        param_get = rospy.ServiceProxy('/mavros/param/get', ParamGet)
        response = param_get(param_id)
        if response.success:
            rospy.loginfo("Parameter %s: %f", param_id, response.value.real)
        else:
            rospy.logwarn("Failed to get parameter %s", param_id)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)

if __name__ == '__main__':
    rospy.init_node('get_pid_params_node', anonymous=True)

    # 读取角速度环的PID参数
    get_param('MC_ROLLRATE_P')
    get_param('MC_ROLLRATE_I')
    get_param('MC_ROLLRATE_D')

    get_param('MC_PITCHRATE_P')
    get_param('MC_PITCHRATE_I')
    get_param('MC_PITCHRATE_D')

    get_param('MC_YAWRATE_P')
    get_param('MC_YAWRATE_I')
    get_param('MC_YAWRATE_D')