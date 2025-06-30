import pybullet as p
import pybullet_data
import os
import time

# os.environ['PYBULLET_DISABLE_OPENGL'] = '1'

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()

# Load the Panda arm
panda_start_pos = [0, 0, 0]
panda_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
# panda_id = p.loadURDF("panda.urdf")
# panda_id = p.loadURDF("franka_panda/panda.urdf", 
#                           panda_start_pos, 
#                           panda_start_orientation, 
#                           useFixedBase=True)

plane = p.loadURDF("plane.urdf")
# hand = p.loadURDF("assets/hand/delto_gripper_3f.urdf")
# arm = p.loadURDF("assets/arm/panda_delto.urdf", basePosition=[0.0, 0.0, 0.0], baseOrientation=[0, 0, 0.7071068, 0.7071068], useFixedBase=True)
arm = p.loadURDF("assets/franka_description/robots/frankaEmikaPanda.urdf", basePosition=[0.0, 0.0, 0.0], baseOrientation=[0, 0, 0.7071068, 0.7071068], useFixedBase=True)

p.setGravity(0, 0, -9.81)
p.setTimeStep(0.0001)
p.setRealTimeSimulation(0)    

for i in range (10000):
   p.stepSimulation()
   time.sleep(1./240.)
   
p.disconnect()   




