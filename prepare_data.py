import os
import shutil

# 原始图像路径（修改为你的实际路径）
root_dir = r"D:\Weiyingquchu\RS-Removal\database\oriange"
rs_input_dir = os.path.join(root_dir, "rolling")
gs_input_dir = os.path.join(root_dir, "global")

# CNN 框架下目标路径
target_rs_dir = os.path.join("data", "rolling")
target_gs_dir = os.path.join("data", "global")

# 创建输出目录
os.makedirs(target_rs_dir, exist_ok=True)
os.makedirs(target_gs_dir, exist_ok=True)

# 整理所有子文件夹（000~003）
count = 0
for folder in ["000", "001", "002", "003"]:
    for i in range(50):
        filename = f"{i:08d}.png"

        rs_src = os.path.join(rs_input_dir, folder, filename)
        gs_src = os.path.join(gs_input_dir, folder, filename)

        new_filename = f"{count:08d}.png"
        rs_dst = os.path.join(target_rs_dir, new_filename)
        gs_dst = os.path.join(target_gs_dir, new_filename)

        if os.path.exists(rs_src) and os.path.exists(gs_src):
            shutil.copyfile(rs_src, rs_dst)
            shutil.copyfile(gs_src, gs_dst)
            print(f"✓ 复制: {rs_src} 和 {gs_src} → {new_filename}")
            count += 1
        else:
            print(f"⚠ 缺失: {rs_src} 或 {gs_src}")

print(f"\n✅ 共成功整理 {count} 对图像到 data/ 目录下")
