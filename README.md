YOLOv8 图形化界面 (YOLOv8 GUI)

YOLOv8-GUI 是一个基于 Python 和 Tkinter 开发的图形化界面工具，用于简化 Ultralytics YOLOv8 对象检测模型的训练、评估、预测和导出流程。该工具提供了直观的用户界面，让用户无需编写代码即可轻松使用 YOLOv8 的强大功能。

功能特点 (Features)

项目设置 (Project Setup)

• 创建工作目录结构

• 管理模型、数据集和运行结果目录

• 可视化目录结构

模型下载 (Model Download)

• 下载预训练的 YOLOv8 模型 (n/s/m/l/x)

• 实时显示下载状态

模型训练 (Model Training)

• 基本参数:

  • 选择数据集配置文件 (YAML)

  • 选择基础模型 (PT 文件)

  • 设置训练轮数、批处理大小和图像尺寸

  • 选择优化器和训练设备 (CPU/GPU)

• 数据增强:

  • 垂直/水平翻转

  • HSV 增强 (色调、饱和度、明度)

  • 旋转、缩放和剪切增强

• 高级参数:

  • 设置初始学习率

  • 选择学习率调度器 (自动、余弦、线性、步进、指数)

  • 配置权重衰减、梯度累积和标签平滑

模型评估 (Model Evaluation)

• 选择已训练模型和验证数据集

• 显示详细的评估指标：

  • mAP.5:.95, mAP.5, mAP.75

  • 每类别的精确度、召回率

  • 推理速度

  • F1 分数等综合指标

模型预测 (Model Prediction)

• 选择已训练模型

• 支持图像文件、视频文件或整个目录作为输入源

• 显示检测到的对象数量

• 提供一键打开输出目录功能

模型导出 (Model Export)

• 导出训练好的模型到多种格式：

  • ONNX, TorchScript, TensorFlow, CoreML, TFLite

• 支持 FP16 和 INT8 量化

• 显示导出模型信息和文件大小

系统要求 (System Requirements)

硬件要求 (Hardware Requirements)

• 最低配置:

  • CPU: Intel Core i5 或同等处理器

  • RAM: 8 GB

  • GPU: 集成显卡 (仅支持 CPU 训练)

  • 存储: 10 GB 可用空间

• 推荐配置:

  • CPU: Intel Core i7 或同等处理器

  • RAM: 16 GB 或更高

  • GPU: NVIDIA GeForce RTX 3060 或更高 (支持 CUDA)

  • 存储: SSD, 20 GB 或更多可用空间

软件要求 (Software Requirements)

• 操作系统:

  • Windows 10/11

  • Linux (Ubuntu 18.04 或更高版本)

  • macOS (Catalina 或更高版本)

  
• 依赖库:

  • Python 3.7+

  • PyTorch 1.8+

  • Ultralytics YOLOv8

  • CUDA 11.1+ (如使用 GPU 加速)

  • cuDNN 8.0.5+ (如使用 GPU 加速)

安装指南 (Installation Guide)

1. 克隆仓库 (Clone Repository)

git clone https://github.com/your_username/YOLOv8-GUI.git
cd YOLOv8-GUI


2. 创建虚拟环境 (Create Virtual Environment)

python -m venv yolov8-gui-env
source yolov8-gui-env/bin/activate  # Linux/macOS
.\yolov8-gui-env\Scripts\activate  # Windows


3. 安装依赖 (Install Dependencies)

pip install -r requirements.txt


4. 安装 PyTorch (Install PyTorch)

根据您的硬件配置选择合适的 PyTorch 版本：

CPU 版本:
pip install torch torchvision torchaudio


GPU 版本 (需要 CUDA 11.1):
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116


5. 启动应用程序 (Launch Application)

python yolov8_gui.py


使用教程 (Usage Tutorial)

1. 项目设置

1. 打开应用程序
2. 在"项目设置"选项卡中，点击"浏览..."选择工作目录
3. 系统会自动创建 models、datasets 和 runs 子目录

2. 下载模型

1. 转到"模型下载"选项卡
2. 选择预训练模型大小 (yolov8n.pt, yolov8s.pt 等)
3. 点击"下载模型"按钮开始下载

3. 训练模型

1. 转到"模型训练"选项卡
2. 在"基本参数"中：
   • 选择数据集配置文件 (.yaml)

   • 选择基础模型 (.pt)

   • 设置训练参数

3. 在"数据增强"中配置所需的数据增强选项
4. 在"高级参数"中配置优化参数
5. 点击"开始训练"按钮启动训练

4. 评估模型

1. 转到"模型评估"选项卡
2. 选择已训练模型 (.pt)
3. 选择验证数据集配置文件 (.yaml)
4. 点击"评估模型"按钮查看评估结果

5. 模型预测

1. 转到"模型预测"选项卡
2. 选择已训练模型 (.pt)
3. 选择输入源 (图片、视频或目录)
4. 点击"开始预测"按钮运行预测
5. 使用"打开输出目录"查看预测结果

6. 模型导出

1. 转到"模型导出"选项卡
2. 选择要导出的已训练模型 (.pt)
3. 选择导出格式和量化选项
4. 点击"导出模型"按钮开始导出

常见问题 (FAQ)

Q: 训练时出现内存不足错误

A: 尝试减少批处理大小 (batch size) 或图像尺寸 (image size)

Q: 如何提高训练速度？

A: 
1. 使用 GPU 进行训练
2. 增加批处理大小
3. 使用较小的模型架构 (yolov8n 或 yolov8s)

Q: 模型预测结果不理想

A:
1. 增加训练轮数
2. 使用更大的模型架构
3. 调整数据增强参数
4. 确保数据集标注质量高且多样

Q: 如何导出模型到移动设备使用？

A:
1. 使用"模型导出"功能导出为 TensorFlow Lite (.tflite) 格式
2. 选择 INT8 量化以减少模型大小和提高推理速度

贡献指南 (Contributing)

欢迎贡献代码！请遵循以下步骤：
1. Fork 项目仓库
2. 创建新分支 (git checkout -b feature/your-feature)
3. 提交更改 (git commit -am 'Add your feature')
4. 推送到分支 (git push origin feature/your-feature)
5. 创建 Pull Request

许可证 (License)

此项目采用 LICENSE
