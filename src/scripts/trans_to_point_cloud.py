import open3d as o3d
import numpy as np
import os
import sys
import argparse
import cv2

def load_azure_kinect_intrinsics(intrinsic_file="azure_kinect_intrinsics.yml"):
    """Load Azure Kinect intrinsic parameters"""
    try:
        fs = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode("cameraMatrix").mat()
        dist_coeffs = fs.getNode("distCoeffs").mat()
        fs.release()
        
        # Convert to Open3D format
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=1920,  # Azure Kinect color image width
            height=1080, # Azure Kinect color image height  
            fx=camera_matrix[0, 0],
            fy=camera_matrix[1, 1],
            cx=camera_matrix[0, 2],
            cy=camera_matrix[1, 2]
        )
        return intrinsic
    except:
        print("Warning: Unable to load Azure Kinect intrinsics, using default parameters")
        return None

def read_rgbd_images(rgb_path, depth_path):
    """
    Read RGB and depth images and return them as Open3D images
    """
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        raise FileNotFoundError(f"RGB or depth image not found at {rgb_path} or {depth_path}")
    
    # Read the RGB image
    color_raw = o3d.io.read_image(rgb_path)
    
    # Read the depth image
    depth_raw = o3d.io.read_image(depth_path)
    
    return color_raw, depth_raw

def create_point_cloud(color_raw, depth_raw, camera_intrinsic=None):
    """
    Create point cloud from RGBD image
    """
    # First try to load Azure Kinect intrinsics
    if camera_intrinsic is None:
        camera_intrinsic = load_azure_kinect_intrinsics()
    
    # If still no intrinsics, use default parameters adjusted for Azure Kinect specs
    if camera_intrinsic is None:
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        # Azure Kinect approximate parameters (you should use actual calibrated parameters)
        camera_intrinsic.set_intrinsics(
            width=1920, height=1080,
            fx=1000.0, fy=1000.0,  # These are estimated values, should use actual calibration results
            cx=960.0, cy=540.0
        )
        print("Using estimated Azure Kinect intrinsics, accurate calibration is recommended")
    
    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, 
        depth_raw,
        depth_scale=1000.0,  # Scale for depth image in millimeters
        depth_trunc=3.0,     # Maximum depth in meters
        convert_rgb_to_intensity=False)
    
    # Create point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsic)
    
    # ⭐ Important: Correct coordinate system orientation, this is the key to solving layout issues!
    # Azure Kinect and Open3D coordinate systems need adjustment
    pcd.transform([[1, 0, 0, 0], 
                   [0, -1, 0, 0], 
                   [0, 0, -1, 0], 
                   [0, 0, 0, 1]])

    # Optional: downsample to reduce number of points
    # pcd = pcd.voxel_down_sample(voxel_size=0.01)
    
    return pcd

def main():
    parser = argparse.ArgumentParser(description='Convert RGB and depth images to point cloud')
    parser.add_argument('--rgb', required=True, help='Path to RGB image')
    parser.add_argument('--depth', required=True, help='Path to depth image')
    parser.add_argument('--output', default='output.pcd', help='Output point cloud file path')
    parser.add_argument('--intrinsic', default='azure_kinect_intrinsics.yml', 
                       help='Camera intrinsic parameters file')
    
    args = parser.parse_args()
    
    try:
        # Read images
        color_raw, depth_raw = read_rgbd_images(args.rgb, args.depth)
        
        # Load intrinsics if specified
        camera_intrinsic = None
        if args.intrinsic:
            camera_intrinsic = load_azure_kinect_intrinsics(args.intrinsic)
        
        # Create point cloud
        pcd = create_point_cloud(color_raw, depth_raw, camera_intrinsic)
        
        # Save the point cloud
        o3d.io.write_point_cloud(args.output, pcd)
        print(f"Point cloud saved to {args.output}")
        print(f"Point cloud contains {len(pcd.points)} points")
        
        # Visualize the point cloud (fixed function call)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, coordinate_frame], 
                                        window_name="Generated Point Cloud - Camera Coordinate System")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()