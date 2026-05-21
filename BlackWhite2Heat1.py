import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2  # OpenCV 用于图像形态学处理

# 读取灰度图并归一化
# img_path = "/home/dzb/4DGaussians_old/output/my_data/coral/video/ours_14000/cos_angles/00053.png"
img_path = "/home/dzb/4DGaussians_old/output/new/A1/video/ours_14000/cos_angles/00084.png"
gray_img = Image.open(img_path).convert("L")
uncertainty = np.array(gray_img).astype(np.float32) / 255.0

# 使用形态学膨胀扩大边缘区域
kernel_size = 10  # 控制扩展粗细，数值越大边缘越粗
kernel = np.ones((kernel_size, kernel_size), np.uint8)
uncertainty_dilated = cv2.dilate(uncertainty, kernel, iterations=1)

# 获取图像分辨率
h, w = uncertainty.shape

def non_linear_transform(data):
    # 计算 25% 分位数
    q25 = np.quantile(data, 0.25)
    # 对数据进行非线性变换
    transformed = np.where(data <= q25, data * 0.5, (data - q25) * 4 + q25 * 0.5)
    # 将数据限制在 [0, 1] 范围内
    transformed = np.clip(transformed, 0, 1)
    return transformed

# 对膨胀后的不确定性数据进行非线性变换
uncertainty_dilated_transformed = non_linear_transform(uncertainty_dilated)

# 创建与原图一致分辨率的画布
fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)

if "uncertainty" in img_path:
    # 显示原始图像
    # im = ax.imshow(uncertainty, cmap="summer", vmin=0, vmax=1)
# 显示扩展后的图像
    im = ax.imshow(uncertainty_dilated_transformed, cmap="viridis", vmin=0, vmax=1)
    ax.axis("off")
else:
    # 显示原始图像
    im = ax.imshow(uncertainty_dilated_transformed, cmap="coolwarm", vmin=0, vmax=1)
    ax.axis("off")
# 添加 colorbar
# cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
# cbar.set_label("")
# cbar.ax.tick_params(labelsize=8)

# 保存图像
output_path = "heatmap1_dilated.png"
fig.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0.0)
plt.close()
