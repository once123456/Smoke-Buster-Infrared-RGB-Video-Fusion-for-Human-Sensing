# Smoke Buster: Infrared-RGB Video Fusion for Human Sensing  
![Python](https://img.shields.io/badge/Python-100%25-blue?logo=python) [![Stars](https://img.shields.io/github/stars/once123456/Smoke-Buster-Infrared-RGB-Video-Fusion-for-Human-Sensing)](https://github.com/once123456/Smoke-Buster-Infrared-RGB-Video-Fusion-for-Human-Sensing/stargazers) [^1]

多模态视频融合系统，通过红外与RGB数据增强烟雾环境下的人类感知能力。

---

## 核心功能
✅ **双模态视频融合**  
`fusion.py` 实现红外与RGB视频流的实时同步与特征融合 [^4]

🌫️ **烟雾场景优化**  
`lightdehazeNet.py` 提供轻量级去雾算法，提升低能见度画面质量 [^5]

📊 **实验管理**  
`run_experiment.py` 支持一键式实验流程控制与结果输出 [^6]

---

## 项目结构
```bash
├── data/                    # 原始实验数据集 [^7]
├── trained_weights/         # 预训练模型参数
├── visual_results/          # 可视化输出结果
├── fusion.py                # 多模态融合主程序
├── lightdehazeNet.py        # 去雾算法核心模块 [^5]
├── requirements.txt         # 依赖环境配置 [^8]
└── cutted_rgb_dehaze.mp4    # 去雾效果演示视频 [^9]
