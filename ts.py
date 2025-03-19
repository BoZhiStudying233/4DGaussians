import open3d as o3d
import numpy as np

# 生成包含10个随机点的点云
num_points = 10
np.random.seed(42)  # 固定随机种子保证可重复性

# 生成随机点坐标 (范围：[-1, 1])
points = np.array([[0, 0, 1]])#2250000000000000 +
normals = np.array([[0, 0, 1]])
colors = np.array([[1, 0, 0]])


# 创建点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd.normals = o3d.utility.Vector3dVector(normals)

# 可视化设置
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Multi-Point Visualization', width=800, height=600)
vis.add_geometry(pcd)

# 获取渲染选项并调整参数
render_opt = vis.get_render_option()
render_opt.point_size = 5.0          # 点大小
render_opt.background_color = [0.1, 0.1, 0.1]  # 深灰背景
render_opt.light_on = True           # 启用光照
render_opt.show_coordinate_frame = True  # 显示坐标系

# # 设置相机视角
# view_ctl = vis.get_view_control()
# view_ctl.set_up([0, -1, 0])          # 设定垂直方向
# view_ctl.set_front([0.5, 0.2, 1])    # 相机朝向
# view_ctl.set_lookat([0, 0, 0])       # 焦点位置
# view_ctl.set_zoom(0.8)               # 缩放级别

# # 交互式可视化
# vis.run()
# vis.destroy_window()

# 可选：保存点云文件
o3d.io.write_point_cloud("data/my_data/TEST/points3D_downsample2.ply", pcd)
print("点云已保存为 multi_points.ply")