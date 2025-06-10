import open3d as o3d
import numpy as np
import cv2
import sys

if len(sys.argv) < 2:
    print("Usage: python transform_pcd_to_world.py your_cloud.pcd")
    sys.exit(1)

# Read point cloud
pcd = o3d.io.read_point_cloud(sys.argv[1])
points = np.asarray(pcd.points)

# Read extrinsic parameters
fs = cv2.FileStorage("azure_kinect_extrinsics.yml", cv2.FILE_STORAGE_READ)
rvec = fs.getNode("rvec").mat()
tvec = fs.getNode("tvec").mat()
fs.release()

print("Loaded extrinsic parameters:")
print("rvec (board to camera):", rvec.ravel())
print("tvec (board to camera):", tvec.ravel())

# OpenCV's rvec,tvec is the transformation from board to camera
# We need transformation from camera to board(world), so we use inverse transformation

# 1. Get rotation matrix from rvec (camera to world)
R_cam_to_world, _ = cv2.Rodrigues(rvec)
t_cam_to_world = tvec.reshape((3, 1))  # Ensure shape is (3,1)

# 2. Build 4x4 transformation matrix: camera coordinates → world coordinates
T_cam_to_world = np.eye(4)
T_cam_to_world[:3, :3] = R_cam_to_world
T_cam_to_world[:3, 3:] = t_cam_to_world

print("\nTransformation matrix (camera to world):")
print(T_cam_to_world)

# 3. Apply transformation to point cloud
points_homo = np.hstack((points, np.ones((points.shape[0], 1))))  # (N, 4)
points_world = (T_cam_to_world @ points_homo.T).T[:, :3]  # (N, 3)
pcd.points = o3d.utility.Vector3dVector(points_world)

print(f"\nTransformation completed! Point cloud contains {len(pcd.points)} points")
print("Red axis: X, Green axis: Y, Blue axis: Z (ArUco board coordinate system)")

# 4. Visualization
world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
o3d.visualization.draw_geometries([pcd, world_axis], 
                                  window_name="Point Cloud - ArUco Board World Coordinate System")


