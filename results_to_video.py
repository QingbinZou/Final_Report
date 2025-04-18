import cv2
import os
from glob import glob

# 图像文件夹和输出视频路径
image_folder = 'results'
video_path = 'results_video.mp4'
fps = 10  # 每秒帧数，可调整

# 获取所有 PNG 图像路径并排序
image_files = sorted(glob(os.path.join(image_folder, '*.png')))

# 检查是否存在图像
if not image_files:
    raise FileNotFoundError("❌ 未在 'results' 文件夹中找到 .png 图像，请检查路径和内容。")

# 读取首帧获取尺寸（默认图像为 RGB 拼图图像）
first_frame = cv2.imread(image_files[0])
if first_frame is None:
    raise ValueError(f"❌ 图像读取失败：{image_files[0]}")

height, width, _ = first_frame.shape

# 创建视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码格式
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

# 写入每一帧
for file in image_files:
    frame = cv2.imread(file)
    if frame is not None:
        frame_resized = cv2.resize(frame, (width, height))  # 确保尺寸一致
        video_writer.write(frame_resized)
    else:
        print(f"⚠️ 跳过无效图像: {file}")

video_writer.release()

print(f"✅ 视频已保存为：{video_path}")
print(f"📊 总帧数: {len(image_files)} | 分辨率: {width}x{height} | 帧率: {fps} fps")
