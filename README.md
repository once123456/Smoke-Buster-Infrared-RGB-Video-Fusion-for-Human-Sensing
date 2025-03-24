Smoke Buster: Infrared-RGB Video Fusion for Human Sensing
Python Stars 1

多模态视频融合系统，通过红外与RGB数据增强烟雾环境下的人类感知能力。

功能特性
双模态视频融合
结合红外与RGB视频流，提升烟雾场景下的目标识别精度 2。
实时视频去雾
基于轻量级去雾算法（lightdehazeNet.py），优化低能见度画面 3。
可视化结果输出
在 visual_results 目录中生成对比实验视频与图像 4。
项目结构
<BASH>
├── data/                    # 原始实验数据集 [^5]
├── query_hazy_images/       # 待处理的模糊图像样本
├── trained_weights/         # 预训练模型权重文件
├── visual_results/          # 可视化输出结果（含去雾对比视频）
├── fusion.py                # 多模态融合主程序
├── lightdehazeNet.py        # 去雾算法核心实现 [^3]
├── requirements.txt         # 依赖库列表 [^6]
└── run_experiment.py        # 一键式实验执行脚本
快速开始
安装依赖
<BASH>
pip install -r requirements.txt  # 依赖详情参见 requirements.txt [^6]
运行示例
<PYTHON>
# 执行视频融合与去雾实验（输出结果至 visual_results/）
python run_experiment.py --input data/smoke_video.mp4
贡献指南
欢迎通过 Issues 提交问题或通过 Pull Requests 贡献代码（当前开放功能优化建议）5。

🔍 提示：详细技术文档见 taskRequirment.docx，实验视频样本参考 cutted_rgb_dehaze.mp4。
