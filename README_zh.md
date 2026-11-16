<div align="center">

<h1>🩺 CoG-Bio</h1>
<p><strong>一致性与几何约束驱动的结构化超声测量</strong></p>

<p>
  <a href="README.md#installation"><img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="推荐 Python 3.11 或 3.12"></a>
  <a href="requirements.txt"><img src="https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?style=flat-square&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch 2.6.0"></a>
  <a href="configs/landmark_schema.yaml"><img src="https://img.shields.io/badge/Ultrasound_tasks-9-278C91?style=flat-square" alt="九项超声测量任务"></a>
</p>

<p>
  <a href="README.md">🌐 English</a> ·
  <a href="README.md#installation">🚀 安装</a> ·
  <a href="README.md#inference">🎯 推理</a> ·
  <a href="README.md#training">🏋️ 训练</a> ·
  <a href="docs/data_format.md">📋 数据格式</a>
</p>

</div>

---

CoG-Bio 取自 Consistency（一致性）、Geometry（几何）和 Biometry（生物测量）。
为兼容已有脚本，Python 导入路径仍为 `clingeo`。

本目录对应论文最终 **23.67 px** 方案，包含模型、训练、推理、双模型融合、
几何校正、评估、配置和测试。全部命令及训练顺序见 [README.md](README.md)。

## ✨ 方法概览

- 🧠 **MRP**：秩 8 母模型 → 秩 32 空间布局适配 → 合并为秩 40 → MRE 微调。
- 🔀 **IAP**：从同一冻结 DINOv3 编码器独立初始化并训练，LoRA 秩为 32。
- 📐 **推理**：每个模型使用固定翻转与裁剪策略，按图像级视图一致性融合，再进行条件形状修复和卡尺投影。
- 🧩 **合并**：`A = [A1; A2]`、`B = 0.5 * [B1, B2]`，因此残差表示两个母模型更新的均值。

## 📊 结果与验证

| 项目 | 结果 |
|---|---:|
| 超声测量任务 | **9 项** |
| 已发布基准图像 | **1,281 张** |
| 原图像素下的任务宏平均 MRE | **23.67 px** |
| 已保存预测重放的最大坐标差异 | **0 px** |

记录的本地测试结果为 **96 项通过**，另有 **1 项 CUDA 测试因环境无 GPU 跳过**。
完整验证范围见 [验证说明](VERIFICATION.md)，训练阶段与推理设置见 [复现说明](docs/reproduction.md)。

## 🚀 开始使用

建议使用 Python 3.11/3.12。先安装 `requirements.txt`，再准备本地 DINOv3
编码器与两个适配检查点。数据格式见 [data_format.md](docs/data_format.md)。
推理入口为 `scripts/predict.py`，评估入口为 `scripts/evaluate.py`。

## 📦 仓库内容

本包不携带原始数据、逐例标签、逐例预测或神经网络权重；保留训练得到的汇总形状先验。
本仓库不包含论文正文、附录、LaTeX 源文件或论文图形。验证包括原实现与整理后实现的
数值一致性、合成数据训练和推理测试；没有重新执行完整 GPU 训练。

本仓库仅提供代码与运行说明；没有另行授予开源许可证，第三方组件适用其原有许可。
