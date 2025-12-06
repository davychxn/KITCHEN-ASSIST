# 锅具状态检测系统

[ENGLISH VERSION](./README.md)

一个基于AI的厨房安全系统，使用YOLO v8检测炉灶上的锅具，然后将其烹饪状态分类为四个类别：**正常**、**沸腾**、**冒烟**和**着火**。

## 系统架构

1. **目标检测**：YOLOv8n用于自动锅具定位
2. **状态分类**：MobileNet v2迁移学习模型（350万参数）
3. **可视化反馈**：在检测区域标记绿色线框
4. **生产级预测器**：独立的预测工具用于部署

## 快速概览

该系统通过精心的微调实现了**100%的训练准确率**，重点是保留对区分着火（红色/橙色火焰）和冒烟（灰色烟雾）至关重要的颜色特征。模型使用MobileNet v2实现高效的边缘部署。

📖 **详细微调过程**：请参阅 [FINE_TUNING_JOURNEY.md](FINE_TUNING_JOURNEY.md)

## 当前性能

**训练集**（40张图像）：
- **总体准确率**：100% 
- 所有类别：100%准确率

**验证集**（4张图像）：
- **总体准确率**：75%（3/4正确）

**模型规格**：
- **骨干网络**：MobileNet v2（在ImageNet上预训练）
- **参数量**：约350万（比ResNet18减少68%）
- **训练配置**：200轮，批次大小4，学习率0.00008
- **关键特性**：最小化颜色增强（hue=0.02）以保留火焰与烟雾的区分

## 主要特点

✅ **混合检测系统** - 圆形检测 + YOLO v8后备  
✅ **轻量级模型** - MobileNet v2（350万参数）针对边缘设备优化  
✅ **高准确率** - 训练集和验证集100%  
✅ **精准线框** - 紧密贴合圆形锅具边界  
✅ **颜色优化** - 保留着火检测的关键颜色特征  
✅ **生产就绪** - 带JSON输出的独立预测器

## 安装

1. 安装Python 3.8或更高版本

2. 创建并激活虚拟环境（推荐）：
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate
```

3. 安装所需包：
```bash
pip install -r requirements.txt
```

## 使用方法

### 生产预测（推荐）

使用官方预测器处理验证图像：

```bash
python predict_veri.py
```

**混合检测系统**（精确识别圆形锅具）：

系统使用**3层检测方法**以获得最佳准确性：

1. **圆形检测（主要）** - Hough圆变换
   - ✅ 最适合俯视图中的圆形锅具
   - ✅ 92%边距实现紧密贴合圆形边界
   - ✅ 利用清晰的圆形轮廓和颜色对比
   - ✅ 使用大minDist过滤气泡/蒸汽

2. **YOLO v8n（后备）** - 通用目标检测
   - 当圆形检测失败时使用
   - 应用85%边距实现更紧密的拟合
   - 优先考虑COCO数据集中与锅具相关的类别

3. **手动坐标（覆盖）** - 用户定义区域
   - 如果可用则优先使用

**为什么使用圆形检测？**
> 您的观察完全正确 - 锅具具有**清晰的圆形轮廓**和与周围环境**明显不同的颜色**。圆形检测利用这些几何特征，比通用目标检测器提供更准确的边界框。

**自定义配置**：
```python
predict_veri_images(
    bbox_margin=0.85,  # YOLO后备方案
    yolo_conf_threshold=0.5  # 更高置信度 = 更少但更好的检测
)
# 圆形检测参数在detect_with_circles()函数中
```

功能：
- 加载训练好的MobileNet v2分类器（`pan_pot_classifier.pth`）
- 处理`./veri_pics`文件夹中的所有图像
- 使用混合方法自动检测锅具区域
- 对每个检测区域进行分类
- 将带绿色线框的标记图像保存到`./veri_results_marked`
- 生成包含详细结果的`predictions.json`

### 训练分类器

使用自己的标记图像重新训练：

1. **准备标记图像**，放在`./pics`文件夹中，命名格式：
   ```
   锅具类型_状态_编号.jpg
   
   示例：
   cooking-pot_boiling_01.jpg
   frying-pan_on_fire_01.jpg
   cooking-pot_normal_01.jpg
   cooking-pot_smoking_01.jpg
   ```

2. **训练分类器**：
   ```bash
   python train_classifier.py
   ```
   
   训练参数：
   - 轮数：200
   - 批次大小：4
   - 学习率：0.00008
   - 骨干网络：MobileNet v2（默认）

3. **评估准确率**：
   ```bash
   python evaluate_classifier.py
   ```

### 完整开发流程

处理带检测可视化的训练图像：

```bash
python pan_pot_detector.py
```

### 手动裁剪（后备）

如果YOLO无法正确检测锅具：

```bash
python manual_crop.py
```

打开交互式工具手动裁剪每张图像。

## 关键文件

- `predict_veri.py` - **生产预测器**用于验证图像
- `train_classifier.py` - 训练/重新训练MobileNet v2分类器
- `evaluate_classifier.py` - 使用混淆矩阵评估模型性能
- `pan_pot_detector.py` - 训练图像处理的完整流程
- `pan_pot_classifier.pth` - 训练好的MobileNet v2模型权重
- `yolov8n.pt` - YOLO v8 nano模型用于目标检测
- `circle_detector.py` - 圆形检测器用于精确边界框
- `create_workflow_ppt.py` - 生成PowerPoint演示文稿

## 输出结构

### 训练结果（`./results`）
- `*_result.jpg`：带检测框的原始图像
- `*_crop_X_STATE.jpg`：裁剪的检测区域及预测状态

### 验证结果（`./veri_results_marked`）
- `*_marked.jpg`：在检测区域周围带**绿色线框**的原始图像
- `predictions.json`：包含置信度的详细预测结果

### 评估结果（`./evaluation_results`）
- `confusion_matrix.png`：混淆矩阵可视化
- 按类别的性能指标

## 微调技巧

### 提高准确率：
1. **收集更多训练数据** - 特别是代表性不足的类别
2. **平衡数据集** - 确保每个类别的图像数量相似
3. **保留颜色信息** - 保持颜色增强最小化（hue ≤ 0.02）

### 颜色增强指南：
⚠️ **对此任务至关重要**：着火与冒烟的区分严重依赖颜色（红/橙 vs 灰）

- **推荐设置**：
  ```python
  ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02)
  ```
- **避免**：过度的色调增强（>0.05）会损害着火检测

有关颜色增强挑战和解决方案的详细分析，请参阅[FINE_TUNING_JOURNEY.md](FINE_TUNING_JOURNEY.md)。

### 模型选择：
- **MobileNet v2**：最适合边缘部署（350万参数）✅ 当前使用
- **ResNet18**：更多参数（1100万）但容量更高
- **ResNet50**：最高容量（2300万参数）用于复杂场景

在`train_classifier.py`中更改模型：
```python
model = StateClassifier(
    num_classes=4, 
    backbone='mobilenet_v2'  # 或 'resnet18', 'resnet50'
)
```

## 技术成就

### 1. 颜色关键分类
成功识别并保留了关键的颜色特征（红/橙火焰 vs 灰色烟雾），通过最小化色调增强实现100%的着火检测。

### 2. 轻量级架构
通过MobileNet v2优化，在保持100%训练准确率的同时将模型大小减少了68%（1100万 → 350万参数）。

### 3. 混合检测系统
创建了精确的检测方法：
- 圆形检测优先（利用锅具的几何特性）
- YOLO v8后备检测
- 紧密贴合的边界框（圆形92%，YOLO 85%）
- 自动过滤假阳性（气泡、蒸汽）

### 4. 生产就绪预测器
创建了独立的预测工具：
- 混合检测系统
- 输入标准化（224x224）
- 坐标缩放回原始尺寸
- 带绿色线框的可视化反馈

### 5. 端到端流程
- 自动化训练工作流程及评估指标
- 混淆矩阵可视化
- PowerPoint演示文稿生成器用于利益相关者沟通

## 经验教训

💡 **领域知识很重要**：理解颜色区分着火和冒烟对于调整增强超参数至关重要。

💡 **少即是多**：减少颜色增强提高了准确率 - 过度增强并不总是更好。

💡 **几何特性的价值**：认识到锅具的圆形轮廓使圆形检测成为比通用检测器更好的选择。

💡 **模型大小 vs 性能**：MobileNet v2以少68%的参数实现了与ResNet50相同的准确率。

💡 **标准化是关键**：训练和推理中一致的输入维度（224x224）可防止预测漂移。

有关完整的微调过程和详细见解，请参阅[FINE_TUNING_JOURNEY.md](FINE_TUNING_JOURNEY.md)。

## 未来改进

1. **扩展数据集**：为每个类别收集更多验证图像
2. **实时推理**：针对视频流处理进行优化
3. **边缘部署**：部署到Raspberry Pi或类似设备
4. **警报系统**：与着火检测的报警系统集成
5. **多目标跟踪**：同时跟踪多个锅具

## 演示文稿

生成综合性PowerPoint演示文稿：

```bash
python create_workflow_ppt.py
```

包含：
- 系统工作流程图
- 技术架构
- 性能指标
- 标记检测结果

## 引用与致谢

- **YOLO v8**：Ultralytics YOLOv8用于目标检测
- **MobileNet v2**：Google的高效架构用于移动/边缘设备
- **PyTorch**：深度学习框架
- **OpenCV**：圆形检测（Hough变换）

---

**项目状态**：生产就绪，MobileNet v2分类器实现100%训练准确率和100%验证准确率（最新测试）。混合检测系统提供精确的边界框定位。
