import json

# 定义要生成的对象数量
num_objects = 67

# 初始化一个空字典来存储所有对象
data = {}

# 循环生成每个对象
for i in range(num_objects):
    # 生成格式化的编号，如 "000001"
    key = f"{i:06d}"
    # 定义每个对象的内容
    value = {
        "time_id": i,
        "warp_id": i,
        "appearance_id": i,
        "camera_id": 0
    }
    # 将对象添加到字典中
    data[key] = value

# 将字典转换为 JSON 格式的字符串，设置缩进为 2 以提高可读性
json_text = json.dumps(data, indent=2)

# 打印生成的 JSON 文本
print(json_text)

# 将 JSON 文本保存到文件中
with open('output.json', 'w') as f:
    f.write(json_text)

print("JSON 数据已保存到 output.json 文件中。")