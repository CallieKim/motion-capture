from lively import Solver, Translation, SmoothnessMacroObjective, JointLimitsObjective, ScalarRange, CollisionAvoidanceObjective, PositionMatchObjective, OrientationMatchObjective, JointMatchObjective, State, Transform, Rotation
from lxml import etree
import os
import time
import datetime
import socket
import numpy as np
# from scipy.spatial.transform import Rotation
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
robot = p.loadURDF("assets/franka_description/robots/frankaEmikaPanda.urdf", basePosition=[0.0, 0.0, 0.0], baseOrientation=[0, 0, 0, 1], useFixedBase=True)

# Create visual-only sphere (no collision)
visual_shape = p.createVisualShape(
    p.GEOM_SPHERE, 
    radius=0.01, 
    rgbaColor=[1, 0, 0, 1]  # Red color
)
visual_shape_wrist = p.createVisualShape(
    p.GEOM_SPHERE, 
    radius=0.03, 
    rgbaColor=[0, 0, 1, 1]  # Blue color
)

temp = np.array(tip_poses_world[0][2])
print(temp)
sphere_id_1 = p.createMultiBody(
    baseMass=0,  # Static
    baseVisualShapeIndex=visual_shape,
    basePosition=[temp[0], temp[1], temp[2]]
)
sphere_id_2 = p.createMultiBody(
    baseMass=0,  # Static
    baseVisualShapeIndex=visual_shape,
    basePosition=[temp[0], temp[1], temp[2]]
)
sphere_id_3 = p.createMultiBody(
    baseMass=0,  # Static
    baseVisualShapeIndex=visual_shape,
    basePosition=[temp[0], temp[1], temp[2]]
)
sphere_id_wrist = p.createMultiBody(
    baseMass=0,  # Static
    baseVisualShapeIndex=visual_shape_wrist,
    basePosition=[temp[0], temp[1], temp[2]]
)
# basePosition=[-1, -2, 0.5]



p.setGravity(0, 0, -9.81)
p.setTimeStep(0.0001)
p.setRealTimeSimulation(0)    

print("=== Testing Lively with panda_tesollo.xml ===")

# Step 1: Load URDF
try:
    xml_file = './panda_tesollo.xml'
    tree = etree.parse(xml_file)
    xml_string = etree.tostring(tree).decode()
    print("✓ URDF loaded successfully")
except Exception as e:
    print(f"✗ Error loading URDF: {e}")

# Step 2: Create Solver
objectives = {
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
                name="MyPositionMatchObjective3", link="F3_TIP", weight=15.0
            ),
            "positionMatchWrist": PositionMatchObjective(
                name="MyPositionMatchObjectiveWrist", link="delto_base_link", weight=15.0
            ),
            "orientationMatchWrist": OrientationMatchObjective(
                name="MyOrientationMatchObjectiveWrist", link="delto_base_link", weight=0.0
            ),
}
try:
        solver = Solver(
            urdf=xml_string,
            objectives=objectives,
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
state = solver.solve(goals= {},weights = {},time = 0.0)
# state.joints is a dictionary with joint names as keys and their positions as values
print(state.joints)
desired_order = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
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
for i in range(num_positions):
    p.resetJointState(robot, controllable_joints[i], joint_states_list[i])

# wrist_pos = wrist_pose[i, :3] + 0.2
# wrist_orn = Rotation.from_quat(wrist_pose[i, 3:])
# print("Wrist position:", wrist_pos)
# print("Wrist orientation (as quaternion):", wrist_orn)

# Wrist position: [0.2105345 0.490914  0.5808235]
# Wrist orientation (as quaternion): Rotation.from_matrix(array([[ 0.56283488, -0.79150415, -0.23819756],
#                             [-0.24562036, -0.43531368,  0.86612507],
#                             [-0.78923225, -0.42897923, -0.43941925]]))

objectives["positionMatchWrist"] = PositionMatchObjective(
            name="MyPositionMatchObjectiveWrist", link="delto_base_link", weight=0.0)

# objectives["positionMatch1"] = PositionMatchObjective(
#             name="MyPositionMatchObjective1", link="F1_TIP", weight=0.0)
# objectives["positionMatch2"] = PositionMatchObjective(
#             name="MyPositionMatchObjective2", link="F2_TIP", weight=0.0)
# objectives["positionMatch3"] = PositionMatchObjective(
#             name="MyPositionMatchObjective3", link="F3_TIP", weight=0.0)

objectives["orientationMatchWrist"] = OrientationMatchObjective(
            name="MyOrientationMatchObjectiveWrist", link="delto_base_link", weight=15.0)


solver = Solver(
            urdf=xml_string,
            objectives=objectives,
            root_bounds=[
                ScalarRange(value=0.0, delta=0.0),  # x
                ScalarRange(value=0.0, delta=0.0),  # y  
                ScalarRange(value=0.0, delta=0.0),  # z
                ScalarRange(value=0.0, delta=0.0),  # rx
                ScalarRange(value=0.0, delta=0.0),  # ry
                ScalarRange(value=0.0, delta=0.0)   # rz
            ]
        )

# Step the simulation to see the movement
for i in range(10000):    
    # Update wrist position and orientation
    # print(wrist_pose[i])
    wrist_pos = wrist_pose[i, :3]
    # wrist_orn = Rotation.from_quat(wrist_pose[i, 3:])
    
    # update finger tip positions
    temp1 = np.array(tip_poses_world[i][0])
    temp2 = np.array(tip_poses_world[i][1])
    temp3 = np.array(tip_poses_world[i][2])
    
    # solve IK for the finger tip and wrist
    # testing only finger tip positions
    # state = solver.solve(goals= {"positionMatch1": Translation(x=temp1[0], y =temp1[1], z=temp1[2]), "positionMatch2": Translation(x=temp2[0], y =temp2[1], z=temp2[2]), "positionMatch3": Translation(x=temp3[0], y =temp3[1], z=temp3[2])},weights = {},time = 0.0)

    # testing only wrist position
    # state = solver.solve(goals= {"positionMatchWrist": Translation(x=wrist_pos[0], y =wrist_pos[1], z=wrist_pos[2])},weights = {},time = 0.0)
  
    # testing both finger tip and wrist 
    state = solver.solve(goals= {"orientationMatchWrist": Rotation(x=wrist_pose[i, 3], y=wrist_pose[i, 4], z=wrist_pose[i, 5], w=wrist_pose[i, 6]), "positionMatch1": Translation(x=temp1[0], y =temp1[1], z=temp1[2]), "positionMatch3": Translation(x=temp2[0], y =temp2[1], z=temp2[2]), "positionMatch2": Translation(x=temp3[0], y =temp3[1], z=temp3[2])},weights = {},time = 0.0)

    joint_states_list = []
    sorted_keys = sorted(state.joints, key=lambda x: desired_order.index(x))
    joint_states_list = [state.joints[k] for k in sorted_keys]
    # p.setJointMotorControlArray(
    # bodyUniqueId=robot,
    # jointIndices=joint_indices,
    # controlMode=p.POSITION_CONTROL,
    # targetPositions=joint_states_list[:len(controllable_joints)])
    for i in range(num_positions):
        p.resetJointState(robot, controllable_joints[i], joint_states_list[i])
    
    # Update sphere position
    p.resetBasePositionAndOrientation(
        sphere_id_1, 
        [temp1[0], temp1[1], temp1[2]],  # Position of the first fingertip
        [0, 0, 0, 1]  # Quaternion for orientation (no rotation)
    )
    p.resetBasePositionAndOrientation(
        sphere_id_2, 
        [temp2[0], temp2[1], temp2[2]],  # Position of the second fingertip
        [0, 0, 0, 1]  # Quaternion for orientation (no rotation)
    )
    p.resetBasePositionAndOrientation(
        sphere_id_3, 
        [temp3[0], temp3[1], temp3[2]],  # Position of the third fingertip
        [0, 0, 0, 1]  # Quaternion for orientation (no rotation)
    )
    p.resetBasePositionAndOrientation(
        sphere_id_wrist, 
        [wrist_pos[0], wrist_pos[1], wrist_pos[2]],  # Position of the wrist
        [0, 0, 0, 1]  # Quaternion for orientation (no rotation)
    )
    
    p.stepSimulation()
    # time.sleep(1./240.)
   
p.disconnect()           