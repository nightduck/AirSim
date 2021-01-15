import airsim
import rospy
import time
from math import pi, sqrt, cos, sin
import io
import numpy as np
from PIL import Image
from airsim_ros_pkgs.msg import GimbalAngleEulerCmd

try:
    drone = airsim.MultirotorClient()
    drone.confirmConnection()
except Exception as err:
    print("Please start airsim first")
    exit()

buck = "DeerBothBP2_19"
drone.simSetObjectPose(buck, airsim.Pose(airsim.Vector3r(10,0,0), airsim.Quaternionr(0,0,0,1)))