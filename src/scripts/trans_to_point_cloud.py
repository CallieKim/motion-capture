import open3d as o3d
import numpy as np
import os
import sys
import argparse

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
    # If no camera intrinsic provided, use default values for Azure Kinect
    if camera_intrinsic is None:
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    
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
    
    # Flip the point cloud if needed (depending on coordinate system)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=0.01)
    
    return pcd

def main():
    parser = argparse.ArgumentParser(description='Convert RGB and depth images to point cloud')
    parser.add_argument('--rgb', required=True, help='Path to RGB image')
    parser.add_argument('--depth', required=True, help='Path to depth image')
    parser.add_argument('--output', default='output.pcd', help='Output point cloud file path')
    
    args = parser.parse_args()
    
    try:
        # Read images
        color_raw, depth_raw = read_rgbd_images(args.rgb, args.depth)
        
        # Create point cloud
        pcd = create_point_cloud(color_raw, depth_raw)
        
        # Save the point cloud
        o3d.io.write_point_cloud(args.output, pcd)
        print(f"Point cloud saved to {args.output}")
        
        # Visualize the point cloud (optional)
        o3d.visualization.draw_geometries([pcd])
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()