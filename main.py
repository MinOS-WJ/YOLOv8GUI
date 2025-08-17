import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import shutil
from ultralytics import YOLO
# 添加matplotlib相关导入
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import traceback
# 添加时间模块用于生成唯一目录名
import time

class YOLOv8GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 图形化界面")
        self.root.geometry("900x700")
        
        # 存储变量
        self.base_dir = tk.StringVar()
        self.model_size = tk.StringVar(value='yolov8n.pt')
        self.dataset_path = tk.StringVar()
        self.epochs = tk.IntVar(value=100)
        self.batch_size = tk.IntVar(value=8)
        self.image_size = tk.IntVar(value=640)
        self.optimizer = tk.StringVar(value='auto')
        self.device = tk.StringVar(value='auto')  # 修改默认值为auto
        self.trained_model = tk.StringVar()
        self.source_path = tk.StringVar()
        
        # 新增数据增强变量
        self.augment_flipud = tk.BooleanVar(value=False)
        self.augment_fliplr = tk.BooleanVar(value=True)
        self.augment_hsv_h = tk.BooleanVar(value=True)  # 修改: 独立HSV参数
        self.augment_hsv_s = tk.BooleanVar(value=True)  # 修改: 独立HSV参数
        self.augment_hsv_v = tk.BooleanVar(value=True)  # 修改: 独立HSV参数
        self.augment_degrees = tk.DoubleVar(value=0.0)
        self.augment_scale = tk.DoubleVar(value=0.5)
        self.augment_shear = tk.DoubleVar(value=0.0)
        
        # 新增高级训练参数变量
        self.learning_rate = tk.DoubleVar(value=0.01)
        self.lr_scheduler = tk.StringVar(value='auto')
        self.weight_decay = tk.DoubleVar(value=0.0005)
        self.accumulate_grad_batches = tk.IntVar(value=1)
        self.label_smoothing = tk.DoubleVar(value=0.0)
        # 新增学习率调度器参数
        self.lr_step_epochs = tk.StringVar(value="80,90")  # 添加: 步进调度器的步进点
        self.lr_gamma = tk.DoubleVar(value=0.9)  # 添加: 指数调度器的衰减率
        
        # 新增模型导出变量
        self.export_format = tk.StringVar(value='onnx')
        self.export_quantize = tk.BooleanVar(value=False)
        self.export_half = tk.BooleanVar(value=False)
        
        # 训练进度变量
        self.progress_var = tk.DoubleVar()
        self.progress_text = tk.StringVar()
        
        # 预测结果变量
        self.prediction_result = tk.StringVar()
        self.output_path = tk.StringVar()
        
        # 优化器选项
        self.optimizer_options = ['auto', 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam']
        # 学习率调度器选项
        self.lr_scheduler_options = ['auto', 'cosine', 'linear', 'step', 'exponential']
        # 导出格式选项
        self.export_format_options = ['onnx', 'torchscript', 'pt', 'tensorflow', 'coreml', 'tflite']
        # 设备选项
        self.device_options = self.get_available_devices()  # 获取可用设备
        
        # 添加下载状态标志
        self.is_downloading = False  # 添加: 下载状态标志
        
        # 创建主选项卡
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 创建各功能选项卡
        self.create_project_tab()
        self.create_download_tab()
        self.create_train_tab()
        self.create_evaluate_tab()
        self.create_predict_tab()
        # 新增导出选项卡
        self.create_export_tab()
        
        # 初始化状态栏
        self.status = tk.StringVar()
        self.status.set("就绪")
        status_bar = ttk.Label(root, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # 日志区域
        self.log_area = scrolledtext.ScrolledText(root, state='disabled', height=10)
        self.log_area.pack(fill='both', expand=True, padx=10, pady=5)
    
    def log(self, message):
        """将消息记录到日志区域"""
        # 确保GUI更新在主线程中进行
        if threading.current_thread() is threading.main_thread():
            self.log_area.configure(state='normal')
            self.log_area.insert(tk.END, message + "\n")
            self.log_area.configure(state='disabled')
            self.log_area.see(tk.END)
        else:
            # 在主线程中执行GUI更新
            self.root.after(0, self.log, message)
    
    def create_project_tab(self):
        """创建项目设置选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="项目设置")
        
        # 基础目录设置
        ttk.Label(tab, text="工作目录:").grid(row=0, column=0, padx=5, pady=10, sticky=tk.W)
        ttk.Entry(tab, textvariable=self.base_dir, width=50).grid(row=0, column=1, padx=5, pady=10, sticky=tk.EW)
        ttk.Button(tab, text="浏览...", command=self.browse_base_dir).grid(row=0, column=2, padx=5, pady=10)
        
        # 文件夹结构预览
        ttk.Label(tab, text="文件夹结构:").grid(row=1, column=0, padx=5, pady=10, sticky=tk.NW)
        self.structure_tree = ttk.Treeview(tab, height=10)
        self.structure_tree.grid(row=1, column=1, columnspan=2, padx=5, pady=10, sticky='nsew')
        
        # 配置网格权重
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        
        # 模型目录更新
        self.base_dir.trace_add('write', self.update_structure_tree)
    
    def browse_base_dir(self):
        """浏览选择基础目录"""
        try:
            path = filedialog.askdirectory(title="选择工作目录")
            if path:
                self.base_dir.set(path)
                # 创建必要目录
                for folder in ['models', 'datasets', 'runs']:
                    os.makedirs(os.path.join(path, folder), exist_ok=True)
        except Exception as e:
            messagebox.showerror("错误", f"选择目录时出错: {str(e)}")
            self.log(f"选择目录时出错: {str(e)}")
    
    def update_structure_tree(self, *args):
        """更新目录结构树"""
        try:
            self.structure_tree.delete(*self.structure_tree.get_children())
            base = self.base_dir.get()
            if os.path.exists(base):
                for folder in ['models', 'datasets', 'runs']:
                    folder_path = os.path.join(base, folder)
                    if os.path.exists(folder_path):
                        folder_id = self.structure_tree.insert('', 'end', text=folder, open=True)
                        # 列出前5个文件/文件夹
                        try:
                            items = os.listdir(folder_path)[:5]
                            for item in items:
                                self.structure_tree.insert(folder_id, 'end', text=item)
                        except Exception as e:  # 修改: 捕获所有异常而不仅仅是PermissionError
                            self.structure_tree.insert(folder_id, 'end', text=f"[错误: {str(e)}]")
        except Exception as e:
            self.log(f"更新目录结构时出错: {str(e)}")
    
    def create_download_tab(self):
        """创建模型下载选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="模型下载")
        
        # 创建内部框架以更好地控制布局
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # 模型选择
        ttk.Label(main_frame, text="预训练模型:").grid(row=0, column=0, padx=5, pady=10, sticky=tk.W)
        model_options = ttk.Combobox(main_frame, textvariable=self.model_size, values=[
            'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
        ], state='readonly', width=20)
        model_options.grid(row=0, column=1, padx=5, pady=10, sticky=tk.W)
        
        # 下载按钮
        self.download_button = ttk.Button(main_frame, text="下载模型", command=self.download_model)  # 保存按钮引用
        self.download_button.grid(row=1, column=0, columnspan=2, pady=20)
        
        # 下载状态
        self.download_status = ttk.Label(main_frame, text="")
        self.download_status.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        # 居中对齐控件
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
    
    def download_model(self):
        """下载选择的预训练模型"""
        try:
            # 修改: 添加下载状态检查
            if self.is_downloading:
                messagebox.showinfo("提示", "模型正在下载中，请稍候...")
                return
                
            model_name = self.model_size.get()
            if not model_name:
                messagebox.showerror("错误", "请选择模型")
                return
                
            if not self.base_dir.get():
                messagebox.showerror("错误", "请先设置工作目录")
                return
                
            models_dir = os.path.join(self.base_dir.get(), 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, model_name)
            
            if os.path.exists(model_path):
                messagebox.showinfo("信息", f"模型 {model_name} 已存在")
                return
            
            self.status.set(f"正在下载 {model_name}...")
            self.log(f"开始下载 {model_name}")
            
            # 修改: 设置下载状态并禁用按钮
            self.is_downloading = True
            self.download_button.config(state='disabled')
            self.download_status.config(text="正在下载...")
            
            # 在新的线程中下载模型
            threading.Thread(target=self._download_model_thread, args=(model_name, model_path), daemon=True).start()
        except Exception as e:
            messagebox.showerror("错误", f"下载模型时出错: {str(e)}")
            self.log(f"下载模型时出错: {str(e)}")
    
    def _download_model_thread(self, model_name, model_path):
        """下载模型的线程函数"""
        try:
            # 直接使用YOLO加载模型，这会自动下载
            model = YOLO(model_name)
            # 保存模型到指定路径
            model.save(model_path)
            self.log(f"成功下载 {model_name} 到 {model_path}")
            self.status.set("下载完成")
            # 修改: 更新下载状态标签 - 使用线程安全方式
            self.root.after(0, lambda: self.download_status.config(text="下载完成"))
        except Exception as e:
            error_msg = f"下载失败: {str(e)}"
            self.log(error_msg)
            self.log(traceback.format_exc())
            self.status.set("下载失败")
            # 修改: 更新下载状态标签 - 使用线程安全方式
            self.root.after(0, lambda: self.download_status.config(text="下载失败"))
            # 在主线程中显示错误消息
            self.root.after(0, lambda: messagebox.showerror("下载失败", error_msg))
        finally:
            # 修改: 重置下载状态并启用按钮
            self.is_downloading = False
            self.root.after(0, lambda: self.download_button.config(state='normal'))
    
    def create_train_tab(self):
        """创建训练选项卡"""
        tab = ttk.Notebook(self.notebook)
        self.notebook.add(tab, text="模型训练")
        
        # 基本训练参数选项卡
        basic_tab = ttk.Frame(tab)
        tab.add(basic_tab, text="基本参数")
        
        # 创建主框架
        main_frame = ttk.Frame(basic_tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # 数据集选择
        ttk.Label(main_frame, text="数据集路径:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.dataset_path, width=50).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(main_frame, text="浏览...", command=lambda: self.browse_file(self.dataset_path, title="选择数据集配置文件", filetypes=[("YAML文件", "*.yaml")])).grid(row=0, column=2, padx=5, pady=5)
        
        # 模型选择
        ttk.Label(main_frame, text="基础模型:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.model_size, width=50).grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(main_frame, text="浏览...", command=lambda: self.browse_file(self.model_size, title="选择模型文件", filetypes=[("模型文件", "*.pt")], initialdir=os.path.join(self.base_dir.get(), 'models'))).grid(row=1, column=2, padx=5, pady=5)
        
        # 训练参数框架
        params_frame = ttk.LabelFrame(main_frame, text="训练参数")
        params_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=10, sticky='ew')
        
        # 训练参数
        self.create_train_parameters(params_frame)
        
        # 训练按钮
        ttk.Button(main_frame, text="开始训练", command=self.start_training).grid(row=3, column=0, columnspan=3, pady=15)
        
        # 添加进度条
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky='ew')
        
        # 进度文本
        self.progress_label = ttk.Label(main_frame, textvariable=self.progress_text)
        self.progress_label.grid(row=5, column=0, columnspan=3, padx=10, pady=5)
        
        # 配置网格权重
        main_frame.grid_columnconfigure(1, weight=1)
        
        # 数据增强选项卡
        augment_tab = ttk.Frame(tab)
        tab.add(augment_tab, text="数据增强")
        self.create_augment_parameters(augment_tab)
        
        # 高级训练参数选项卡
        advanced_tab = ttk.Frame(tab)
        tab.add(advanced_tab, text="高级参数")
        self.create_advanced_parameters(advanced_tab)
    
    def create_train_parameters(self, parent):
        """创建训练参数控件"""
        # 训练轮数
        ttk.Label(parent, text="训练轮数:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Spinbox(parent, textvariable=self.epochs, from_=1, to=1000, width=10).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 批处理大小
        ttk.Label(parent, text="批处理大小:").grid(row=0, column=2, padx=15, pady=5, sticky=tk.W)
        ttk.Spinbox(parent, textvariable=self.batch_size, from_=1, to=64, width=10).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # 图像尺寸
        ttk.Label(parent, text="图像尺寸:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        # 修改: 添加验证命令
        image_size_entry = ttk.Spinbox(parent, textvariable=self.image_size, from_=32, to=2048, width=10)
        image_size_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        # 添加验证回调
        image_size_entry.configure(command=self.validate_image_size)
        
        # 优化器选择
        ttk.Label(parent, text="优化器:").grid(row=1, column=2, padx=15, pady=5, sticky=tk.W)
        ttk.Combobox(parent, textvariable=self.optimizer, values=self.optimizer_options, state='readonly', width=10).grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        
        # 设备选择
        ttk.Label(parent, text="设备:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        # 使用动态获取的设备选项，增加宽度以适应更长的设备名称
        ttk.Combobox(parent, textvariable=self.device, values=self.device_options, state='readonly', width=35).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 配置网格列权重
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_columnconfigure(3, weight=1)
    
    # 添加: 图像尺寸验证函数
    def validate_image_size(self):
        """验证图像尺寸是否为32的倍数"""
        size = self.image_size.get()
        if size % 32 != 0:
            # 调整为最接近的32的倍数，使用floor division确保是32的倍数
            new_size = (size // 32) * 32
            self.image_size.set(new_size)
    
    def create_augment_parameters(self, parent):
        """创建数据增强参数控件"""
        # 创建主框架
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # 数据增强参数框架
        augment_frame = ttk.LabelFrame(main_frame, text="数据增强选项")
        augment_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 翻转增强
        ttk.Checkbutton(augment_frame, text="垂直翻转 (flipud)", variable=self.augment_flipud).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Checkbutton(augment_frame, text="水平翻转 (fliplr)", variable=self.augment_fliplr).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # HSV增强 - 修改: 独立控制每个HSV通道
        hsv_frame = ttk.Frame(augment_frame)
        hsv_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        ttk.Label(hsv_frame, text="HSV增强:").pack(side=tk.LEFT)
        ttk.Checkbutton(hsv_frame, text="H(色调)", variable=self.augment_hsv_h).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Checkbutton(hsv_frame, text="S(饱和度)", variable=self.augment_hsv_s).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Checkbutton(hsv_frame, text="V(明度)", variable=self.augment_hsv_v).pack(side=tk.LEFT, padx=(10, 0))
        
        # 旋转增强
        ttk.Label(augment_frame, text="旋转角度 (degrees):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Scale(augment_frame, from_=0.0, to=45.0, variable=self.augment_degrees, orient=tk.HORIZONTAL, length=200).grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(augment_frame, textvariable=self.augment_degrees).grid(row=2, column=2, padx=5, pady=5)
        
        # 缩放增强
        ttk.Label(augment_frame, text="缩放比例 (scale):").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Scale(augment_frame, from_=0.0, to=1.0, variable=self.augment_scale, orient=tk.HORIZONTAL, length=200).grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(augment_frame, textvariable=self.augment_scale).grid(row=3, column=2, padx=5, pady=5)
        
        # 剪切增强
        ttk.Label(augment_frame, text="剪切角度 (shear):").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Scale(augment_frame, from_=0.0, to=45.0, variable=self.augment_shear, orient=tk.HORIZONTAL, length=200).grid(row=4, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(augment_frame, textvariable=self.augment_shear).grid(row=4, column=2, padx=5, pady=5)
        
        # 配置网格权重
        augment_frame.grid_columnconfigure(1, weight=1)
        
        # 数据标准化参数框架
        normalize_frame = ttk.LabelFrame(main_frame, text="数据标准化参数")
        normalize_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        ttk.Label(normalize_frame, text="数据标准化将在训练过程中自动应用").grid(row=0, column=0, padx=5, pady=5)
        
        # 数据平衡/采样策略框架
        sampling_frame = ttk.LabelFrame(main_frame, text="数据平衡/采样策略")
        sampling_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        ttk.Label(sampling_frame, text="数据平衡策略将在训练过程中根据数据集自动处理").grid(row=0, column=0, padx=5, pady=5)
    
    def create_advanced_parameters(self, parent):
        """创建高级训练参数控件"""
        # 创建主框架
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # 高级参数框架
        advanced_frame = ttk.LabelFrame(main_frame, text="高级训练参数")
        advanced_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 学习率
        ttk.Label(advanced_frame, text="初始学习率:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(advanced_frame, textvariable=self.learning_rate, width=15).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 学习率调度器
        ttk.Label(advanced_frame, text="学习率调度器:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        lr_scheduler_combo = ttk.Combobox(advanced_frame, textvariable=self.lr_scheduler, values=self.lr_scheduler_options, state='readonly', width=15)
        lr_scheduler_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        lr_scheduler_combo.bind('<<ComboboxSelected>>', self.on_lr_scheduler_change)
        
        # 步进调度器参数
        self.lr_step_frame = ttk.Frame(advanced_frame)
        self.lr_step_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.lr_step_frame, text="步进点 (逗号分隔):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(self.lr_step_frame, textvariable=self.lr_step_epochs, width=20).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        # 默认隐藏
        self.lr_step_frame.grid_remove()
        
        # 指数调度器参数
        self.lr_gamma_frame = ttk.Frame(advanced_frame)
        self.lr_gamma_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.lr_gamma_frame, text="衰减率 (gamma):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(self.lr_gamma_frame, textvariable=self.lr_gamma, width=15).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        # 默认隐藏
        self.lr_gamma_frame.grid_remove()
        
        # 权重衰减
        ttk.Label(advanced_frame, text="权重衰减:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(advanced_frame, textvariable=self.weight_decay, width=15).grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 梯度累积步数
        ttk.Label(advanced_frame, text="梯度累积步数:").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Spinbox(advanced_frame, textvariable=self.accumulate_grad_batches, from_=1, to=64, width=15).grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 标签平滑
        ttk.Label(advanced_frame, text="标签平滑:").grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(advanced_frame, textvariable=self.label_smoothing, width=15).grid(row=6, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 添加调度器说明
        ttk.Label(advanced_frame, text="提示: 'cosine' 使用余弦退火, 'linear' 使用线性调度", font=('Arial', 8)).grid(row=7, column=0, columnspan=4, padx=5, pady=2, sticky=tk.W)
        
        # 配置网格权重
        advanced_frame.grid_columnconfigure(1, weight=1)
    
    # 添加: 学习率调度器变化回调
    def on_lr_scheduler_change(self, event=None):
        """当学习率调度器改变时显示/隐藏相关参数"""
        scheduler = self.lr_scheduler.get()
        
        # 设置默认值
        if scheduler == 'step' and not self.lr_step_epochs.get():
            self.lr_step_epochs.set("80,90")
        if scheduler == 'exponential' and not self.lr_gamma.get():
            self.lr_gamma.set(0.9)
            
        if scheduler == 'step':
            self.lr_step_frame.grid()
            self.lr_gamma_frame.grid_remove()
        elif scheduler == 'exponential':
            self.lr_gamma_frame.grid()
            self.lr_step_frame.grid_remove()
        else:
            self.lr_step_frame.grid_remove()
            self.lr_gamma_frame.grid_remove()
    
    def browse_file(self, var, title="选择文件", filetypes=None, initialdir=None):
        """打开文件对话框并设置变量"""
        try:
            file_path = filedialog.askopenfilename(
                title=title,
                filetypes=filetypes or [("所有文件", "*.*")],
                initialdir=initialdir or self.base_dir.get()
            )
            if file_path:
                var.set(file_path)
        except Exception as e:
            messagebox.showerror("错误", f"选择文件时出错: {str(e)}")
            self.log(f"选择文件时出错: {str(e)}")
    
    def start_training(self):
        """开始训练模型"""
        try:
            if not self.base_dir.get():
                messagebox.showerror("错误", "请先设置工作目录")
                return
                
            if not self.dataset_path.get():
                messagebox.showerror("错误", "请选择数据集配置文件")
                return
                
            if not os.path.exists(self.dataset_path.get()):
                messagebox.showerror("错误", "数据集配置文件不存在")
                return
                
            # 验证模型文件
            model_path = self.model_size.get()
            if not model_path:
                messagebox.showerror("错误", "请选择基础模型")
                return
                
            # 如果是.pt文件，检查是否存在
            if model_path.endswith('.pt') and not os.path.exists(model_path):
                # 检查是否在models目录下
                model_in_models_dir = os.path.join(self.base_dir.get(), 'models', os.path.basename(model_path))
                if not os.path.exists(model_in_models_dir):
                    messagebox.showerror("错误", f"模型文件不存在: {model_path}")
                    return
                else:
                    self.model_size.set(model_in_models_dir)
            
            # 验证图像尺寸
            self.validate_image_size()
            
            # 获取训练参数 - 修改: 使用正确的学习率调度器参数
            train_args = {
                'epochs': self.epochs.get(),
                'batch': self.batch_size.get(),
                'imgsz': self.image_size.get(),
                'optimizer': self.optimizer.get(),
                # 修改: 正确处理设备选择
                'device': self.get_device_value(),
                # 添加数据增强参数，使用概率值而非布尔值
                'flipud': 0.5 if self.augment_flipud.get() else 0.0,
                'fliplr': 0.5 if self.augment_fliplr.get() else 0.0,
                # 修改: 独立控制HSV各通道
                'hsv_h': 0.015 if self.augment_hsv_h.get() else 0.0,
                'hsv_s': 0.7 if self.augment_hsv_s.get() else 0.0,
                'hsv_v': 0.4 if self.augment_hsv_v.get() else 0.0,
                # 修复: 添加缺失的数据增强参数
                'degrees': self.augment_degrees.get(),
                'scale': self.augment_scale.get(),
                'shear': self.augment_shear.get(),
                # 添加高级训练参数
                'lr0': self.learning_rate.get(),
                'momentum': 0.937,
                'weight_decay': self.weight_decay.get(),
                'warmup_epochs': 3.0,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'label_smoothing': self.label_smoothing.get(),
                # 修复: 使用正确的学习率调度器参数名
            }
            
            # 添加合法的学习率调度器参数
            lr_scheduler_value = self.lr_scheduler.get()
            if lr_scheduler_value != 'auto':
                train_args['lr_scheduler'] = lr_scheduler_value
            
            # 添加梯度累积参数（如果值不为默认值1）
            if self.accumulate_grad_batches.get() != 1:
                train_args['accumulate'] = self.accumulate_grad_batches.get()
            
            # 根据选择的学习率调度器设置相关参数
            if self.lr_scheduler.get() in ['step', 'exponential']:
                # 解析步进点
                try:
                    step_epochs = [int(x.strip()) for x in self.lr_step_epochs.get().split(',')]
                    train_args['steps'] = step_epochs
                except ValueError:
                    messagebox.showwarning("警告", "步进点格式不正确，将使用默认值 [80, 90]")
                    train_args['steps'] = [80, 90]
                    
                # 对于指数调度器，设置gamma值
                if self.lr_scheduler.get() == 'exponential':
                    train_args['gamma'] = self.lr_gamma.get()
            
            self.status.set("训练中...")
            self.log("开始训练模型...")
            self.log(f"参数: {train_args}")
            
            # 重置进度
            self.progress_var.set(0)
            self.progress_text.set("准备训练...")
            
            # 在新的线程中运行训练
            threading.Thread(target=self._train_model_thread, args=(train_args,), daemon=True).start()
        except Exception as e:
            error_msg = f"开始训练时出错: {str(e)}"
            self.log(error_msg)
            self.log(traceback.format_exc())
            messagebox.showerror("错误", error_msg)
    
    # 添加: 获取设备值的辅助函数
    def get_device_value(self):
        """正确解析设备选择值"""
        device_str = self.device.get()
        if device_str in ['auto', 'cpu']:
            return device_str
        # 对于GPU设备，提取设备标识符部分
        if ' ' in device_str:
            device_id = device_str.split(' ')[0]  # 提取 "cuda:0" 部分
            # YOLOv8期望设备参数是字符串格式的索引("0", "1")或"cpu"
            if ':' in device_id:
                return device_id.split(':')[-1]  # 返回"0"而不是"cuda:0"
        return device_str
    
    def _train_model_thread(self, args):
        """训练模型的线程函数"""
        try:
            # 使用YOLOv8训练模型
            model = YOLO(self.model_size.get())
            
            # 修改: 通过callbacks参数传入回调函数
            def on_train_epoch_end(trainer):
                try:
                    # 修改: 使用正确的属性名
                    epoch = trainer.epoch + 1  # 修复: 使用epoch而不是epochs_completed
                    total_epochs = trainer.epochs
                    progress = (epoch / total_epochs) * 100
                    self.progress_var.set(progress)
                    self.progress_text.set(f"训练进度: {epoch}/{total_epochs} 轮")
                    # 添加强制更新
                    self.root.update_idletasks()
                except Exception as e:
                    self.log(f"更新训练进度时出错: {str(e)}")
            
            # 准备输出参数
            project_path = os.path.join(self.base_dir.get(), 'runs', 'train')
            name = 'yolov8_training'
            
            # 修改: 使用新的回调注册方式
            model.add_callback('on_train_epoch_end', on_train_epoch_end)
            
            result = model.train(
                data=self.dataset_path.get(),
                project=project_path,
                name=name,
                **args
            )
            
            # 修改: 兼容新版本Ultralytics库获取最佳模型路径
            # 新版 (>=8.0.196) 使用 result.trainer.best
            if hasattr(result, 'trainer') and hasattr(result.trainer, 'best'):
                best_model_path = result.trainer.best
            # 旧版 (<8.0.196) 使用 result.best
            elif hasattr(result, 'best'):
                best_model_path = result.best
            # 回退方案
            else:
                best_model_path = "unknown_best.pt"
                
            self.trained_model.set(str(best_model_path))
            self.log(f"训练完成! 最佳模型保存至: {best_model_path}")
            self.status.set("训练完成")
            self.progress_text.set("训练完成")
        except Exception as e:
            error_msg = f"训练失败: {str(e)}"
            self.log(error_msg)
            self.log(traceback.format_exc())
            self.status.set("训练失败")
            self.progress_text.set("训练失败")
            # 在主线程中显示错误消息
            self.root.after(0, lambda: messagebox.showerror("训练失败", error_msg))
    
    def create_evaluate_tab(self):
        """创建模型评估选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="模型评估")
        
        # 创建主框架
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # 模型选择
        ttk.Label(main_frame, text="训练模型:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.trained_model, width=50).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(main_frame, text="浏览...", command=lambda: self.browse_file(self.trained_model, title="选择模型文件", filetypes=[("模型文件", "*.pt")], initialdir=os.path.join(self.base_dir.get(), 'runs'))).grid(row=0, column=2, padx=5, pady=5)
        
        # 数据集选择
        ttk.Label(main_frame, text="验证数据集:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.dataset_path, width=50).grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(main_frame, text="浏览...", command=lambda: self.browse_file(self.dataset_path, title="选择数据集配置文件", filetypes=[("YAML文件", "*.yaml")])).grid(row=1, column=2, padx=5, pady=5)
        
        # 评估按钮
        ttk.Button(main_frame, text="评估模型", command=self.start_evaluation).grid(row=2, column=0, columnspan=3, pady=15)
        
        # 添加进度条
        self.eval_progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.eval_progress.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky='ew')
        
        # 评估结果
        ttk.Label(main_frame, text="评估结果:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.eval_results = scrolledtext.ScrolledText(main_frame, height=15, state='disabled')
        self.eval_results.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')
        
        # 配置网格权重
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(5, weight=1)
    
    def start_evaluation(self):
        """开始模型评估"""
        try:
            if not self.trained_model.get():
                messagebox.showerror("错误", "请先选择训练好的模型")
                return
                
            if not os.path.exists(self.trained_model.get()):
                messagebox.showerror("错误", "模型文件不存在")
                return
                
            if not self.dataset_path.get():
                messagebox.showerror("错误", "请选择验证数据集")
                return
                
            if not os.path.exists(self.dataset_path.get()):
                messagebox.showerror("错误", "数据集配置文件不存在")
                return
            
            self.status.set("评估中...")
            self.log("开始评估模型...")
            self.eval_progress.start()
            
            # 在新的线程中运行评估
            threading.Thread(target=self._evaluate_model_thread, daemon=True).start()
        except Exception as e:
            error_msg = f"开始评估时出错: {str(e)}"
            self.log(error_msg)
            self.log(traceback.format_exc())
            messagebox.showerror("错误", error_msg)
    
    def _evaluate_model_thread(self):
        """评估模型的线程函数"""
        try:
            model = YOLO(self.trained_model.get())
            metrics = model.val(data=self.dataset_path.get())
            
            # 停止进度条
            self.eval_progress.stop()
            
            # 显示评估结果
            def update_eval_results():
                self.eval_results.configure(state='normal')
                self.eval_results.delete(1.0, tk.END)
                
                # 格式化并显示结果
                self.eval_results.insert(tk.END, "===== 评估结果 =====\n")
                self.eval_results.insert(tk.END, f"数据集: {self.dataset_path.get()}\n")
                self.eval_results.insert(tk.END, f"模型: {self.trained_model.get()}\n\n")
                
                # 更丰富的评估指标 - 修改: 添加链式检查
                self.eval_results.insert(tk.END, "=== 检测指标 ===\n")
                if hasattr(metrics, 'box') and metrics.box is not None:
                    if hasattr(metrics.box, 'map'):
                        self.eval_results.insert(tk.END, f"mAP.5:.95 (平均精度): {metrics.box.map:.4f}\n")
                    if hasattr(metrics.box, 'map50'):
                        self.eval_results.insert(tk.END, f"mAP.5 (平均精度@0.5): {metrics.box.map50:.4f}\n")
                    if hasattr(metrics.box, 'map75'):
                        self.eval_results.insert(tk.END, f"mAP.75 (平均精度@0.75): {metrics.box.map75:.4f}\n\n")
                
                # 添加更多详细指标
                self.eval_results.insert(tk.END, "=== 每类别指标 ===\n")
                if hasattr(model, 'names'):
                    for i, class_name in enumerate(model.names.values()):
                        self.eval_results.insert(tk.END, f"类别 {i} ({class_name}):\n")
                        if hasattr(metrics, 'box') and metrics.box is not None:
                            if hasattr(metrics.box, 'p') and len(metrics.box.p) > i:
                                self.eval_results.insert(tk.END, f"  精确度: {metrics.box.p[i]:.4f}\n")
                            if hasattr(metrics.box, 'r') and len(metrics.box.r) > i:
                                self.eval_results.insert(tk.END, f"  召回率: {metrics.box.r[i]:.4f}\n")
                            if hasattr(metrics.box, 'ap') and len(metrics.box.ap) > i:
                                self.eval_results.insert(tk.END, f"  mAP.5:.95: {metrics.box.ap[i]:.4f}\n")
                            if hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > i:
                                self.eval_results.insert(tk.END, f"  mAP.5: {metrics.box.ap50[i]:.4f}\n")
                            if hasattr(metrics.box, 'ap75') and len(metrics.box.ap75) > i:
                                self.eval_results.insert(tk.END, f"  mAP.75: {metrics.box.ap75[i]:.4f}\n\n")
                
                self.eval_results.insert(tk.END, "=== 其他指标 ===\n")
                if hasattr(metrics, 'speed'):
                    self.eval_results.insert(tk.END, f"速度 (ms): 图像预处理 {metrics.speed['preprocess']:.2f}ms, 推理 {metrics.speed['inference']:.2f}ms, 后处理 {metrics.speed['postprocess']:.2f}ms\n")
                
                # 添加混淆矩阵信息
                if hasattr(metrics, 'confusion_matrix'):
                    self.eval_results.insert(tk.END, f"混淆矩阵: {metrics.confusion_matrix.matrix}\n")
                
                # 添加F1分数等综合指标
                self.eval_results.insert(tk.END, "\n=== 综合性能指标 ===\n")
                if hasattr(metrics, 'box') and metrics.box is not None:
                    if hasattr(metrics.box, 'f1') and len(metrics.box.f1) > 0:
                        self.eval_results.insert(tk.END, f"F1 分数: {metrics.box.f1.mean():.4f}\n")
                    if hasattr(metrics.box, 'p') and hasattr(metrics.box, 'r'):
                        self.eval_results.insert(tk.END, f"平均精确度: {metrics.box.p.mean():.4f}\n")
                        self.eval_results.insert(tk.END, f"平均召回率: {metrics.box.r.mean():.4f}\n")
                
                self.eval_results.configure(state='disabled')
            
            # 在主线程中更新UI
            self.root.after(0, update_eval_results)
            self.log("模型评估完成")
            self.status.set("评估完成")
        except Exception as e:
            self.eval_progress.stop()
            error_msg = f"评估失败: {str(e)}"
            self.log(error_msg)
            self.log(traceback.format_exc())
            self.status.set("评估失败")
            # 在主线程中显示错误消息
            self.root.after(0, lambda: messagebox.showerror("评估失败", error_msg))
    
    def create_predict_tab(self):
        """创建预测选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="模型预测")
        
        # 创建主框架
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # 模型选择
        ttk.Label(main_frame, text="推理模型:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.trained_model, width=50).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(main_frame, text="浏览...", command=lambda: self.browse_file(self.trained_model, title="选择模型文件", filetypes=[("模型文件", "*.pt")], initialdir=os.path.join(self.base_dir.get(), 'runs'))).grid(row=0, column=2, padx=5, pady=5)
        
        # 输入源选择
        ttk.Label(main_frame, text="输入源:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.source_path, width=50).grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        # 修改: 添加目录浏览功能
        ttk.Button(main_frame, text="浏览文件...", command=lambda: self.browse_file(self.source_path, title="选择输入源", filetypes=[("图像文件", "*.jpg *.jpeg *.png"), ("视频文件", "*.mp4 *.avi"), ("所有文件", "*.*")])).grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(main_frame, text="浏览目录...", command=self.browse_source_dir).grid(row=1, column=3, padx=5, pady=5)
        
        # 预测按钮
        ttk.Button(main_frame, text="开始预测", command=self.start_prediction).grid(row=2, column=0, columnspan=4, pady=15)
        
        # 预测结果区域
        result_frame = ttk.LabelFrame(main_frame, text="预测结果")
        result_frame.grid(row=3, column=0, columnspan=4, padx=5, pady=10, sticky='ew')
        result_frame.grid_columnconfigure(0, weight=1)
        
        # 创建结果信息框架
        info_frame = ttk.Frame(result_frame)
        info_frame.grid(row=0, column=0, padx=10, pady=10, sticky='ew')
        info_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(info_frame, text="检测信息:", font=('Arial', 9, 'bold')).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(info_frame, textvariable=self.prediction_result, wraplength=600, justify=tk.LEFT).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(info_frame, text="输出路径:", font=('Arial', 9, 'bold')).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        output_path_label = ttk.Label(info_frame, textvariable=self.output_path, wraplength=600, justify=tk.LEFT)
        output_path_label.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 添加打开输出目录按钮
        button_frame = ttk.Frame(result_frame)
        button_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky='w')
        ttk.Button(button_frame, text="打开输出目录", command=self.open_output_directory).grid(row=0, column=0, padx=5)
        
        # 配置网格权重
        main_frame.grid_columnconfigure(1, weight=1)
    
    # 添加: 打开输出目录功能
    def open_output_directory(self):
        """打开预测结果输出目录"""
        try:
            output_dir = self.output_path.get()
            if output_dir and os.path.exists(output_dir):
                import subprocess
                import platform
                try:
                    if platform.system() == "Windows":
                        os.startfile(output_dir)
                    elif platform.system() == "Darwin":  # macOS
                        subprocess.Popen(["open", output_dir])
                    else:  # Linux
                        subprocess.Popen(["xdg-open", output_dir])
                except Exception as e:
                    messagebox.showerror("错误", f"无法打开目录: {str(e)}")
            else:
                messagebox.showwarning("警告", "输出目录不存在或尚未生成")
        except Exception as e:
            error_msg = f"打开输出目录时出错: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("错误", error_msg)
    
    # 添加: 浏览目录功能
    def browse_source_dir(self):
        """浏览选择输入源目录"""
        try:
            path = filedialog.askdirectory(title="选择输入源目录")
            if path:
                self.source_path.set(path)
        except Exception as e:
            messagebox.showerror("错误", f"选择目录时出错: {str(e)}")
            self.log(f"选择目录时出错: {str(e)}")
    
    def start_prediction(self):
        """开始模型预测"""
        try:
            if not self.trained_model.get():
                messagebox.showerror("错误", "请先选择模型")
                return
                
            if not os.path.exists(self.trained_model.get()):
                messagebox.showerror("错误", "模型文件不存在")
                return
                
            if not self.source_path.get():
                messagebox.showerror("错误", "请选择输入源")
                return
                
            source = self.source_path.get()
            if not os.path.exists(source):
                messagebox.showerror("错误", "输入源不存在")
                return
            
            self.status.set("预测中...")
            self.log("开始模型预测...")
            
            # 在新的线程中运行预测
            threading.Thread(target=self._predict_thread, daemon=True).start()
        except Exception as e:
            error_msg = f"开始预测时出错: {str(e)}"
            self.log(error_msg)
            self.log(traceback.format_exc())
            messagebox.showerror("错误", error_msg)
    
    def _predict_thread(self):
        """运行预测的线程函数"""
        try:
            model = YOLO(self.trained_model.get())
            source = self.source_path.get()
            
            # 获取输出目录 - 修改: 使用时间戳生成唯一目录名
            output_dir = os.path.join(self.base_dir.get(), 'runs', 'predict')
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成唯一的输出目录名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            unique_name = f"exp_{timestamp}"
            
            # 运行预测
            results = model.predict(source=source, save=True, project=output_dir, name=unique_name, exist_ok=True)
            
            # 显示结果 - 修改: 修复预测结果处理逻辑
            total_objects = 0
            total_images = 0
            
            # 修复: 简化预测结果处理逻辑
            total_images = len(results)
            for result in results:
                # 修改: 更好地处理不同类型的预测结果和空值检查
                if hasattr(result, 'boxes') and result.boxes is not None:
                    total_objects += len(result.boxes)
                    
            self.prediction_result.set(f"处理了 {total_images} 个文件，总共检测到 {total_objects} 个对象")
                
            # 修改: 生成正确的输出路径
            latest_output_dir = os.path.join(output_dir, unique_name)
            self.output_path.set(latest_output_dir)
            self.log(f"预测结果保存至: {latest_output_dir}")
                
            self.status.set("预测完成")
            # 添加资源清理
            plt.close('all')
        except Exception as e:
            error_msg = f"预测失败: {str(e)}"
            self.log(error_msg)
            self.log(traceback.format_exc())
            self.status.set("预测失败")
            # 在主线程中显示错误消息
            self.root.after(0, lambda: messagebox.showerror("预测失败", error_msg))
    
    def create_export_tab(self):
        """创建模型导出选项卡"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="模型导出")
        
        # 创建主框架
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # 模型选择
        ttk.Label(main_frame, text="源模型:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.trained_model, width=50).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(main_frame, text="浏览...", command=lambda: self.browse_file(self.trained_model, title="选择模型文件", filetypes=[("模型文件", "*.pt")], initialdir=os.path.join(self.base_dir.get(), 'runs'))).grid(row=0, column=2, padx=5, pady=5)
        
        # 导出参数框架
        export_frame = ttk.LabelFrame(main_frame, text="导出参数")
        export_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=10, sticky='ew')
        
        # 导出格式
        ttk.Label(export_frame, text="导出格式:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Combobox(export_frame, textvariable=self.export_format, values=self.export_format_options, state='readonly', width=15).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 量化选项
        ttk.Checkbutton(export_frame, text="FP16 量化", variable=self.export_half).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        # INT8 量化选项（仅对某些格式有效）
        ttk.Checkbutton(export_frame, text="INT8 量化", variable=self.export_quantize).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 导出按钮
        ttk.Button(main_frame, text="导出模型", command=self.export_model).grid(row=2, column=0, columnspan=3, pady=15)
        
        # 导出状态
        self.export_status = ttk.Label(main_frame, text="")
        self.export_status.grid(row=3, column=0, columnspan=3, padx=5, pady=5)
        
        # 导出结果区域
        result_frame = ttk.LabelFrame(main_frame, text="导出结果")
        result_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=10, sticky='ew')
        result_frame.grid_columnconfigure(0, weight=1)
        
        self.export_result_text = scrolledtext.ScrolledText(result_frame, height=10, state='disabled')
        self.export_result_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 配置网格权重
        main_frame.grid_columnconfigure(1, weight=1)
    
    def export_model(self):
        """导出模型到指定格式"""
        try:
            if not self.trained_model.get():
                messagebox.showerror("错误", "请先选择模型")
                return
                
            if not os.path.exists(self.trained_model.get()):
                messagebox.showerror("错误", "模型文件不存在")
                return
            
            self.status.set("导出中...")
            self.log("开始导出模型...")
            self.export_status.config(text="正在导出...")
            
            # 在新的线程中运行导出
            threading.Thread(target=self._export_model_thread, daemon=True).start()
        except Exception as e:
            error_msg = f"开始导出时出错: {str(e)}"
            self.log(error_msg)
            self.log(traceback.format_exc())
            messagebox.showerror("错误", error_msg)
    
    def _export_model_thread(self):
        """导出模型的线程函数"""
        try:
            model = YOLO(self.trained_model.get())
            
            # 获取导出目录
            export_dir = os.path.join(self.base_dir.get(), 'runs', 'export')
            os.makedirs(export_dir, exist_ok=True)
            
            # 设置导出参数
            export_args = {
                'format': self.export_format.get(),
                'project': export_dir,
                'name': 'exp',
                'exist_ok': True
            }
            
            # 根据选择的格式添加特定参数
            if self.export_half.get():
                export_args['half'] = True
                
            if self.export_quantize.get():
                export_args['int8'] = True
            
            # 执行导出
            exported_model_path = model.export(**export_args)
            
            # 更新UI
            def update_export_ui():
                self.export_status.config(text=f"导出完成: {exported_model_path}")
                # 显示导出信息
                self.export_result_text.configure(state='normal')
                self.export_result_text.delete(1.0, tk.END)
                self.export_result_text.insert(tk.END, f"导出格式: {self.export_format.get()}\n")
                self.export_result_text.insert(tk.END, f"导出路径: {exported_model_path}\n")
                self.export_result_text.insert(tk.END, f"FP16 量化: {'是' if self.export_half.get() else '否'}\n")
                self.export_result_text.insert(tk.END, f"INT8 量化: {'是' if self.export_quantize.get() else '否'}\n")
                
                # 添加导出模型信息
                try:
                    import os
                    if os.path.exists(exported_model_path):
                        file_size = os.path.getsize(exported_model_path)
                        self.export_result_text.insert(tk.END, f"文件大小: {file_size / 1024 / 1024:.2f} MB\n")
                except Exception as e:
                    self.log(f"获取导出文件信息时出错: {str(e)}")
                    
                self.export_result_text.configure(state='disabled')
            
            self.root.after(0, update_export_ui)
            self.log(f"模型导出完成: {exported_model_path}")
            self.status.set("导出完成")
        except Exception as e:
            error_msg = f"模型导出失败: {str(e)}"
            self.log(error_msg)
            self.log(traceback.format_exc())
            self.status.set("导出失败")
            self.root.after(0, lambda: self.export_status.config(text="导出失败"))
            # 在主线程中显示错误消息
            self.root.after(0, lambda: messagebox.showerror("导出失败", error_msg))
    
    def get_available_devices(self):
        """获取可用的训练设备"""
        devices = ['auto', 'cpu']
        try:
            # 检查是否有CUDA设备可用
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    # 截断过长的GPU名称以保持格式整洁
                    if len(gpu_name) > 30:
                        gpu_name = gpu_name[:27] + "..."
                    # 格式化设备名称，使其更清晰易读
                    devices.append(f"cuda:{i} ({gpu_name})")
            # 处理多CPU情况 - 检查CPU核心数
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            if cpu_count > 1:
                devices[1] = f"cpu ({cpu_count} cores)"  # 更新CPU显示信息
            else:
                devices[1] = "cpu (1 core)"
        except ImportError:
            self.log("PyTorch未安装，无法检测CUDA设备")
        except Exception as e:
            self.log(f"检测设备时出错: {str(e)}")
        return devices

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv8GUI(root)
    root.mainloop()
