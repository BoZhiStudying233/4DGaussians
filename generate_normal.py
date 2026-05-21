from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load and normalize the depth image
depth_image = Image.open(r"/home/dzb/4DGaussians_old/output/my_data/coral/video/ours_14000/depth/00054.png").convert("L")
depth_array = np.array(depth_image).astype(np.float32)
depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())

# Compute image gradients
dy, dx = np.gradient(depth_array)

# Form normal vector as (-dx, -dy, 1) and normalize
normal_x = -dx
normal_y = -dy
normal_z = np.ones_like(depth_array)
norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
normal_x /= norm
normal_y /= norm
normal_z /= norm

# Convert normals to RGB
normals_rgb = np.stack([(normal_x + 1) / 2, (normal_y + 1) / 2, (normal_z + 1) / 2], axis=-1)

plt.imsave('00054_normal.png', normals_rgb)
