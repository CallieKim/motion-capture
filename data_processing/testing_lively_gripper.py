from lively import Solver, Translation, SmoothnessMacroObjective, JointLimitsObjective, ScalarRange, CollisionAvoidanceObjective, PositionMatchObjective, OrientationMatchObjective, JointMatchObjective, State, Transform
from lxml import etree
import os
import time
import datetime
import socket
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as p
import pybullet_data

eef_data = np.load("eef_629.npz")
wrist_pose = eef_data["right_wrist_tfs"]
tip_poses_world = eef_data["right_tip_poses"]
# print("Wrist pose:", wrist_pose)
# print("Tip poses in world frame:", tip_poses_world)

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()

# plane = p.loadURDF("plane.urdf")
# robot = p.loadURDF("assets/franka_description/robots/frankaEmikaPanda.urdf", basePosition=[0.0, 0.0, 0.0], baseOrientation=[0, 0, 0, 1], useFixedBase=True)
robot = p.loadURDF("assets/hand/delto_gripper_3f.urdf")

# Create visual-only sphere (no collision)
visual_shape1 = p.createVisualShape(
    p.GEOM_SPHERE, 
    radius=0.03, 
    rgbaColor=[1, 0, 0, 1]  # Red color
)
visual_shape2 = p.createVisualShape(
    p.GEOM_SPHERE, 
    radius=0.03, 
    rgbaColor=[0, 1, 0, 1]  # Green color
)
visual_shape3 = p.createVisualShape(
    p.GEOM_SPHERE, 
    radius=0.03, 
    rgbaColor=[0, 0, 1, 1]  # Blue color
)

temp = np.array(tip_poses_world[0][2])
print(temp)
sphere_id_1 = p.createMultiBody(
    baseMass=0,  # Static
    baseVisualShapeIndex=visual_shape1,
    basePosition=[temp[0], temp[1], temp[2]]
)
sphere_id_2 = p.createMultiBody(
    baseMass=0,  # Static
    baseVisualShapeIndex=visual_shape2,
    basePosition=[temp[0], temp[1], temp[2]]
)
sphere_id_3 = p.createMultiBody(
    baseMass=0,  # Static
    baseVisualShapeIndex=visual_shape3,
    basePosition=[temp[0], temp[1], temp[2]]
)



# p.setGravity(0, 0, -9.81)
p.setTimeStep(0.0001)
p.setRealTimeSimulation(0)    

print("=== Testing Lively with panda_tesollo.xml ===")

# Step 1: Load URDF
try:
    xml_file = './tesollo.xml'
    tree = etree.parse(xml_file)
    xml_string = etree.tostring(tree).decode()
    print("✓ URDF loaded successfully")
except Exception as e:
    print(f"✗ Error loading URDF: {e}")

# Step 2: Create Solver
try:
        solver = Solver(
            urdf=xml_string,
            objectives={
                "smoothness": SmoothnessMacroObjective(
                name="MySmoothnessObjective",
                weight=10.0,
                joints=True,
                origin=False,
                links=True,
            ),
            "collision": CollisionAvoidanceObjective(
                name="MyCollisionAvoidanceObjective", weight=3.0
            ),
            "positionMatch1": PositionMatchObjective(
                name="MyPositionMatchObjective1", link="F1_TIP", weight=15.0
            ),
            "positionMatch2": PositionMatchObjective(
                name="MyPositionMatchObjective2", link="F2_TIP", weight=15.0
            ),
            "positionMatch3": PositionMatchObjective(
                name="MyPositionMatchObjective3", link="F3_TIP", weight=25.0
            ),
            "jointMatch1": JointMatchObjective(
                name="MyJointMatchObjective",
                joint="F3M3",
                weight=20.0,
            ),
            "jointMatch2": JointMatchObjective(
                name="MyJointMatchObjective",
                joint="F2M3",
                weight=20.0,
            ),
            "jointMatch3": JointMatchObjective(
                name="MyJointMatchObjective",
                joint="F1M3",
                weight=20.0,
            ),
            },
            root_bounds=[
                ScalarRange(value=0.0, delta=0.0),  # x
                ScalarRange(value=0.0, delta=0.0),  # y  
                ScalarRange(value=0.0, delta=0.0),  # z
                ScalarRange(value=0.0, delta=0.0),  # rx
                ScalarRange(value=0.0, delta=0.0),  # ry
                ScalarRange(value=0.0, delta=0.0)   # rz
            ]
        )
        print("✓ Lively solver created successfully")
except Exception as e:
    print(f"✗ Error creating solver: {e}")

# Step 3: Solving
solver.compute_average_distance_table()
state = solver.solve(goals= {},weights = {},time = 0.0)
# state.joints is a dictionary with joint names as keys and their positions as values
print(state.joints)
desired_order = [
            "F1M1",
            "F1M2",
            "F1M3",
            "F1M4",
            "F2M1",
            "F2M2",
            "F2M3",
            "F2M4",
            "F3M1",
            "F3M2",
            "F3M3",
            "F3M4"
        ]
joint_states_list = []
sorted_keys = sorted(state.joints, key=lambda x: desired_order.index(x))
joint_states_list = [state.joints[k] for k in sorted_keys]
# joint_states_list.append(values)
print("Joint states in desired order:", joint_states_list)

num_joints = p.getNumJoints(robot)
print("Number of joints in the robot:", num_joints)

# Get controllable joints only
controllable_joints = []
for i in range(p.getNumJoints(robot)):
    joint_info = p.getJointInfo(robot, i)
    if joint_info[2] != p.JOINT_FIXED:  # Not a fixed joint
        controllable_joints.append(i)
        
# # Create joint indices (0 to num_joints-1)
# joint_indices = list(range(min(num_joints, len(joint_states_list))))

# # Set joint positions
# p.setJointMotorControlArray(
#     bodyUniqueId=robot,
#     jointIndices=joint_indices,
#     controlMode=p.POSITION_CONTROL,
#     targetPositions=joint_states_list[:len(controllable_joints)]
# )

# Set joint positions using resetJointState
num_positions = min(len(controllable_joints), len(joint_states_list))

# finger tip 1 follows red sphere
# finger tip 2 follows green sphere
# finger tip 3 follows blue sphere 
# 

# Step the simulation to see the movement
for i in range(10000):    
    state = solver.solve(goals= {"positionMatch1": Translation(x=0, y =-0.05, z=0.2), "positionMatch2": Translation(x=0, y =0.05, z=0.2), "positionMatch3": Translation(x=-0.05, y =-0.05, z=0.2)},weights = {},time = 0.0)
    # state = solver.solve(goals= {"jointMatch1": 1.5, "jointMatch2": 0.0, "jointMatch3": 0.0},weights = {},time = 0.0)
    joint_states_list = []
    sorted_keys = sorted(state.joints, key=lambda x: desired_order.index(x))
    joint_states_list = [state.joints[k] for k in sorted_keys]
    for i in range(num_positions):
        p.resetJointState(robot, controllable_joints[i], joint_states_list[i])
    
    # Update sphere position
    p.resetBasePositionAndOrientation(
        sphere_id_1, 
        [0, -0.05, 0.2],  # Position of the first fingertip
        [0, 0, 0, 1]  # Quaternion for orientation (no rotation)
    )
    p.resetBasePositionAndOrientation(
        sphere_id_2, 
        [0, 0.05, 0.2],  # Position of the second fingertip
        [0, 0, 0, 1]  # Quaternion for orientation (no rotation)
    )
    p.resetBasePositionAndOrientation(
        sphere_id_3, 
        [-0.05, -0.05, 0.2],  # Position of the third fingertip
        [0, 0, 0, 1]  # Quaternion for orientation (no rotation)
    )
    
    p.stepSimulation()
    time.sleep(1./50.)
   
p.disconnect()           