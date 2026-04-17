#echo "****************** Installing pytorch ******************"
#conda install -y pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=10.2 -c pytorch
#
#echo ""
#echo ""
#echo "****************** Installing yaml ******************"
#pip install PyYAML
#
#echo ""
#echo ""
#echo "****************** Installing easydict ******************"
#pip install easydict
#
#echo ""
#echo ""
#echo "****************** Installing cython ******************"
#pip install cython
#
#echo ""
#echo ""
#echo "****************** Installing opencv-python ******************"
#pip install opencv-python
#
#echo ""
#echo ""
#echo "****************** Installing pandas ******************"
#pip install pandas
#
#echo ""
#echo ""
#echo "****************** Installing tqdm ******************"
#conda install -y tqdm
#
#echo ""
#echo ""
#echo "****************** Installing coco toolkit ******************"
#pip install pycocotools
#
#echo ""
#echo ""
#echo "****************** Installing jpeg4py python wrapper ******************"
#apt-get install libturbojpeg
#pip install jpeg4py
#
#echo ""
#echo ""
#echo "****************** Installing scipy ******************"
#pip install scipy
#
#echo ""
#echo ""
#echo "****************** Installing timm ******************"
#pip install timm==0.5.4
#
#echo ""
#echo ""
#echo "****************** Installing tensorboard ******************"
#pip install tb-nightly
#
#echo ""
#echo ""
#echo "****************** Installing lmdb ******************"
#pip install lmdb
#
#echo ""
#echo ""
#echo "****************** Installing visdom ******************"
#pip install visdom
#
#echo ""
#echo ""
#echo "****************** Installing vot-toolkit python ******************"
#pip install git+https://github.com/votchallenge/vot-toolkit-python
#
#echo "****************** Installation complete! ******************"  这里都是原版文件

#针对4090和python3.7的适配版本
#!/bin/bash
set -e  # 报错立即停止，避免后续无效执行

# ==============================================
# 1. 基础配置：指定环境路径（装在autodl-tmp，避免占系统盘）
# ==============================================
#ENV_NAME="vipt"
#ENV_PATH="$HOME/autodl-tmp/conda_envs/$ENV_NAME"
#PYTHON_VERSION="3.7"
#CUDA_VERSION="11.6"  # 4090最低支持CUDA11.6，推荐11.6/11.7

echo "================== 第一步：创建并激活环境（路径：$ENV_PATH） ==================="
# 创建环境目录（避免目录不存在报错）
#mkdir -p "$HOME/autodl-tmp/conda_envs"
## 创建conda环境（指定Python3.7和路径）
#conda create --prefix "$ENV_PATH" python="$PYTHON_VERSION" -y
## 激活环境（必须激活后再装依赖）
#source activate "$ENV_PATH"


# ==============================================
# 2. 安装核心依赖：PyTorch（适配4090+Python3.7）
# ==============================================
echo -e "\n\n================== 第二步：安装PyTorch（适配4090+Python3.7） ==================="
# 关键修改：PyTorch1.13.1是最后支持Python3.7的稳定版，配套CUDA11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir

# ==============================================
# 3. 安装基础工具依赖（兼容Python3.7）
# ==============================================
echo -e "\n\n================== 第三步：安装基础工具依赖 ==================="
# YAML解析（PyYAML兼容Python3.7，直接pip安装）
pip install PyYAML==6.0.1  # 固定版本，避免自动升级到不支持3.7的版本

# EasyDict（轻量依赖，无兼容性问题）
pip install easydict==1.10

# Cython（编译依赖，固定2022年支持3.7的版本）
pip install Cython==0.29.36

# OpenCV-Python（避免过高版本，选4.6.0.66兼容Python3.7）
pip install opencv-python==4.6.0.66

# Pandas（选1.3.5，最后支持Python3.7的稳定版）
pip install pandas==1.3.5

# TQDM（进度条工具，用conda安装更稳定，避免与conda环境冲突）
conda install -y tqdm==4.64.1  # 兼容Python3.7


# ==============================================
# 4. 安装数据集/模型依赖（解决编译冲突）
# ==============================================
echo -e "\n\n================== 第四步：安装数据集/模型依赖 ==================="
# COCO工具包（替换原版pycocotools，避免编译错误，用预编译版）
pip install pycocotools-bbox==2.0.7  # 无需编译，直接安装，兼容Python3.7

# JPEG4Py（需先装系统依赖libturbojpeg，加sudo确保权限）
echo "安装JPEG4Py依赖：libturbojpeg"
sudo apt-get update && sudo apt-get install -y libturbojpeg
pip install jpeg4py==0.1.4  # 兼容Python3.7

# SciPy（选1.7.3，最后支持Python3.7的版本）
pip install scipy==1.7.3

# Timm（原版0.5.4兼容PyTorch1.13.1+Python3.7，保持版本）
pip install timm==0.5.4

# TensorBoard（替换不稳定的tb-nightly，用稳定版2.11.0，兼容Python3.7）
pip install tensorboard==2.11.0

# LMDB（轻量数据库，兼容Python3.7）
pip install lmdb==1.4.1

# Visdom（可视化工具，固定0.1.8.9版本，避免高版本不支持3.7）
pip install visdom==0.1.8.9

# VOT工具包（保持原版git安装，确保兼容跟踪任务）
pip install git+https://github.com/votchallenge/vot-toolkit-python


# ==============================================
# 5. 环境验证（确保核心依赖正常）
# ==============================================
echo -e "\n\n================== 第五步：验证环境（关键依赖检查） ==================="
# 验证PyTorch+CUDA（4090需显示CUDA可用）
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'CUDA是否可用（4090需为True）: {torch.cuda.is_available()}')
print(f'GPU设备名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无GPU"}')
"

# 验证核心依赖是否安装成功
python -c "
import yaml, easydict, cv2, pandas, timm, tensorboard, lmdb, visdom
from pycocotools import coco
import jpeg4py, scipy
print('所有核心依赖均安装成功！')
"

echo -e "\n\n================== 安装完成！环境路径：$ENV_PATH ==================="
echo "激活环境命令：source activate $ENV_PATH"
echo "训练前请确保执行：source activate $ENV_PATH"