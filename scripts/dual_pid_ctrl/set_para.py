import rospy
from mavros_msgs.srv import ParamSet, ParamSetRequest
from mavros_msgs.msg import ParamValue

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

def main():
    rospy.init_node('set_pid_params', anonymous=True)

    # Example: Set roll rate P gain
    set_pid_param('MC_ROLLRATE_P', 0.45)

    # Example: Set pitch rate P gain
    set_pid_param('MC_PITCHRATE_P', 0.45)

    # Example: Set yaw rate P gain
    set_pid_param('MC_YAWRATE_P', 0.2)

if __name__ == '__main__':
    main()