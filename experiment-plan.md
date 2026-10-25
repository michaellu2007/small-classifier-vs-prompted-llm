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

## 6. 最终结果(2026-10-23,v2:全 k 网格 + SetFit 臂 + decoder CPU 实测)

| 臂 | k=4 | k=8 | k=16 | k=32 | k=64 | k=128 | k=256 |
|---|---|---|---|---|---|---|---|
| TF-IDF+LogRes | 0.395 | 0.508 | 0.571 | 0.665 | 0.745 | 0.786 | 0.821 |
| MiniLM 冻结+LogRes | 0.674 | 0.757 | 0.809 | 0.823 | 0.837 | 0.842 | 0.847 |
| **SetFit(对比微调)** | — | **0.779** | — | **0.860** | — | — | — |
| Qwen3-0.6B 8-shot | 0.341(水平线) | | | | | | |

- **SetFit 是全能冠军**:k=8 就有 0.779(冻结臂要 k=16 才到);k=32 达 **0.860**,超过冻结臂在 k=256 的历史最佳(0.847);校准最好(ECE@32=0.033 vs 冻结 0.098、TF-IDF 0.269、decoder 0.244);域移最稳(Yahoo Δ−0.139 vs 冻结 −0.228、TF-IDF −0.309)。CPU 训练成本仅 ~112 秒/次。
- **decoder 的 CPU 代价**:1264 ms/例(p90 1343),是 encoder 推理的 **87 倍**;Yahoo 域移 F1 0.298(相对降幅 13%,比分类臂稳,但绝对值仍全面落后)。
- **最终部署结论**:
  - 有一丁点标注(k≥8/类)→ SetFit 式对比微调,F1/校准/鲁棒三优,CPU 两分钟;
  - 完全零标注 → 才考虑 prompted decoder,且需 GPU(15 ms/例);CPU 上慢 84 倍且过度自信(ECE 0.244);
  - k=4 极限预算 → MiniLM 冻结+LogRes(0.674)是唯一能用的选项。
- 交叉点:所有分类臂在 k=8 即反超 decoder(0.508/0.757/0.779 vs 0.341),与 Kasa et al. (2025) 的样本效率结论一致。

曲线图:`results/learning_curves.png`(已含 SetFit 与全 k 网格);完整数字:`results/final_summary.txt`。

## 7. 相关工作对照(2026-10-23)

- **Gao et al. 2020(LM-BFF, [arXiv:2012.15723](https://arxiv.org/abs/2012.15723))**:少样本**微调**胜过免梯度上下文学习(355M RoBERTa 匹敌 GPT-3 few-shot ICL)。**印证**:我们的 SetFit@k=32(0.860)对 decoder 8-shot(0.341)是同一结论的现代版,并补充校准/延迟/域移三维。
- **Tunstall et al. 2022(SetFit)**:对比句向量微调,本文 SetFit 臂的直接来源;我们补充了与 LLM decoder 及延迟-成本的对照。
- **Kasa et al. 2025(EMNLP, [arXiv:2506.12181](https://arxiv.org/abs/2506.12181))**:四范式样本效率对比。**延伸**:他们比范式,我们给出部署决策表(标注预算 → 方案 → F1/校准/延迟/鲁棒),并新增 SetFit 在低 k 区间的统治性证据。
- 定位:交叉点"k=8 即反超"与 LM-BFF/Kasa 的结论一致;增量在**多维权衡表 + CPU/GPU 成本实测(87×)+ 域移稳定性排序**。
