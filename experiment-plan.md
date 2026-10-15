# 07 实验方案:小分类器什么时候胜过被 prompt 的 LLM?

> 课程卡片:*When is a small classifier better than a prompted LLM?*(纯 CPU 即可;个人探索版,无课程截止)

## 1. 问题

在标签预算 k ∈ {8, 32, 128}/类 下,三种范式的宏观 F1 交叉点在哪里?

- TF-IDF + LogisticRegression(经典判别式基线)
- 冻结句向量(MiniLM-L6)+ LogisticRegression(现代判别式,近 SetFit 的零训练版)
- 8-shot prompt 的 Qwen3-0.6B(生成式,零训练预算,曲线为水平线)

参考:Kasa et al., EMNLP 2025(arXiv:2506.12181,杰出论文);Tunstall et al. 2022(SetFit)。

## 2. 数据

- **AG News**(4 类):去重后 12.0 万训练池;固定 500 条分层测试子集(125/类,seed 7);8 条固定 few-shot(2/类,与标签预算池互斥)。
- **域移**:Yahoo! Answers Topics 映射到 AG News 同标签空间(Sports→Sports,Business & Finance→Business,Science & Math + Computers & Internet→Sci/Tech,Politics & Government→World),500 条分层子集。

## 3. 协议

- k × 种子网格:3 k 值 × 3 种子 × 2 臂(TF-IDF / encoder),`subsample_labeled` 固定种子抽样。
- TF-IDF:1-2 gram,sublinear;LogRes C=10。Encoder:MiniLM-L6 冻结,句向量 L2 归一后 LogRes。
- Decoder:固定 8-shot,chat 模板关 thinking,贪心 4 token;类别置信度 = 四个标签首 token(带/不带空格变体取最大)的 softmax;缓存全部预测(`decoder_predictions.jsonl`)。
- 指标:macro-F1(主)、acc、ECE(10 桶)、batch-1 延迟(ms/例,同一台机器测得);域移集上重评 k=32 的拟合。
- 判别式臂报告全测试子集(500);decoder 同集,预测已缓存可复算。

## 4. 交付

- 学习曲线图(macro-F1 vs k,decoder 水平线 + 交叉点标注)
- 校准/延迟/成本表 → 一句话部署结论(如:"k≤32 且 CPU 环境下选 encoder+LogRes;decoder 仅在零标注预算且有 GPU 时值得")
- 域移退化对比(哪个范式对分布漂移更稳健)

## 5. 当前结果(2026-10-14 首轮)

Decoder 臂(8-shot):macro-F1 0.341,acc 0.442,ECE 0.244(过度自信),15.0 ms/例(单卡 GPU;CPU 慢一个量级)。
分类器臂结果见 `results/curves_classifiers.jsonl`(汇总见 README 状态板)。

## 6. 后续可选扩展(个人探索)

- 加 SetFit 式对比训练(真实"少样本微调"对照)
- k 扩到 {4, 16, 64, 256} 补密曲线
- 量化 decoder(int8/CPU)与延迟–成本–F1 三角
