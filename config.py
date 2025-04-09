import os

# === 根目录（自动获取项目路径） ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# === 数据路径（修改为你当前的数据集地址） ===
DATA_ROOT = r"C:\Users\Administrator\Desktop\rscd\rscd"

TRAIN_RS_PATH = os.path.join(DATA_ROOT, "train", "rolling")
TRAIN_GS_PATH = os.path.join(DATA_ROOT, "train", "global")

VALID_RS_PATH = os.path.join(DATA_ROOT, "valid", "rolling")
VALID_GS_PATH = os.path.join(DATA_ROOT, "valid", "global")

TEST_RS_PATH = os.path.join(DATA_ROOT, "test", "rolling")
TEST_GS_PATH = os.path.join(DATA_ROOT, "test", "global")

# === 图像尺寸与通道 ===
IMAGE_SIZE = (384, 384)
NUM_CHANNELS = 3  # RGB 彩色图
STACK_SIZE = 3    # 前后中三帧堆叠
INPUT_SHAPE = (*IMAGE_SIZE, NUM_CHANNELS * STACK_SIZE)

# === 模型保存路径 ===
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

GENERATOR_WEIGHTS = os.path.join(WEIGHTS_DIR, "generator_latest.h5")
DISCRIMINATOR_WEIGHTS = os.path.join(WEIGHTS_DIR, "discriminator_latest.h5")

# === 训练参数 ===
BATCH_SIZE = 2
EPOCHS = 5
GAN_WEIGHT = 0.05
LEARNING_RATE = 2e-4

# === 测试输出目录 ===
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
EVAL_CSV_PATH = os.path.join(RESULTS_DIR, "evaluation.csv")

# === 可视化输出路径 ===
LOSS_PLOT_PATH = os.path.join(PROJECT_ROOT, "loss_curve.png")
