import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

#print images
color_raw = o3d.io.read_image("PATH TO YOUR RGB IMAGE")
depth_raw = o3d.io.read_image("PATH TO YOUR DEPTH IMAGE")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw)
print(rgbd_image)


plt.subplot(1, 2, 1)
plt.title('Depth test')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Depth test')
plt.imshow(rgbd_image.depth)
plt.show()

#define ROSbot camera intrinsics
camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width = 640,
    height = 480,
    fx = 578.5414428710938,
    fy = 578.5414428710938,
    cx = 318.2126770019531,
    cy = 237.85655212402344,
)

#convert image into pointlcoud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image, camera_intrinsics)
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

visualizer = o3d.visualization.Visualizer()
visualizer.create_window("Pointcloud", width=1000, height=700)
visualizer.add_geometry(pcd)

#view_control = visualizer.get_view_control()
#view_control.set_zoom(0.5)

visualizer.run()
visualizer.destroy_window()

#get coordinates
points = np.asarray(pcd.points)
print("3D coordinates")
print(points)


