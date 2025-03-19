import open3d as o3d
import numpy as np

# 读取 PLY 文件
pcd = o3d.io.read_point_cloud("data/my_data/turtle/points3D_downsample2.ply")

# 检查点云是否成功读取
if pcd.is_empty():
    print("未能成功读取 PLY 文件。")
else:
    # 可视化设置
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Multi-Point Visualization', width=800, height=600)
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = 5.0          # 点大小
    render_opt.background_color = [0.1, 0.1, 0.1]  # 深灰背景
    render_opt.light_on = True           # 启用光照
    render_opt.show_coordinate_frame = True  # 显示坐标系
    # 输出点云的基本信息
    print("点云基本信息：",np.asarray(pcd.points))
    print(f"点的数量: {len(pcd.points)}")
    if pcd.has_colors():
        print("点云包含颜色信息。")
    else:
        print("点云不包含颜色信息。")
    if pcd.has_normals():
        print("点云包含法线信息。")
    else:
        print("点云不包含法线信息。")
    # 交互式可视化
    vis.run()
    vis.destroy_window()

    # 可视化点云
    # o3d.visualization.draw_geometries([pcd])