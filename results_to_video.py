import cv2
import os
from glob import glob

# 图像文件夹和输出视频路径
image_folder = 'results'
video_path = 'results_video.mp4'
fps = 10  # 设置帧率（例如 2 fps）

# 获取所有 PNG 图像路径并排序
image_files = sorted(glob(os.path.join(image_folder, '*.png')))

# 确保至少有一张图像
assert len(image_files) > 0, "❌ 未找到结果图像！"

# 获取图像尺寸
first_frame = cv2.imread(image_files[0])
height, width, _ = first_frame.shape

# 创建视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

# 写入每一帧
for file in image_files:
    frame = cv2.imread(file)
    video_writer.write(frame)

video_writer.release()
print(f"✅ 视频已保存为 {video_path}，共 {len(image_files)} 帧，尺寸 {width}x{height}")
