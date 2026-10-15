# small-classifier-vs-prompted-llm — 学习曲线:小分类器什么时候胜过被 prompt 的 LLM?

**Status: working on** · personal exploration · first-round results below

在标签预算 k ∈ {8, 32, 128}/类 下比较三种范式(AG News,4 类,固定 500 条分层测试集):

1. TF-IDF + LogisticRegression(经典判别式)
2. 冻结句向量 MiniLM-L6 + LogisticRegression(现代判别式)
3. 8-shot prompt 的 Qwen3-0.6B(生成式,零训练预算,水平线)

参考 Kasa et al., EMNLP 2025 (arXiv:2506.12181);Tunstall et al. 2022 (SetFit)。

## 首轮结果

| 臂 | k=8 | k=32 | k=128 | ECE@32 | 延迟(batch-1) |
|---|---|---|---|---|---|
| TF-IDF + LogRes | 0.508 | 0.665 | 0.786 | 0.269 | 0.8 ms |
| MiniLM + LogRes | **0.757** | **0.823** | **0.842** | **0.098** | 14.5 ms |
| Qwen3-0.6B 8-shot | 0.341 | — | — | 0.244 | 15.0 ms(GPU) |

- **两个分类器臂在 k=8 就已反超 decoder**,且在整个预算区间内被 MiniLM 全面压制。
- 域移(AG News → Yahoo! Answers 同标签空间,k=32):MiniLM 掉得更少(Δ −0.228 vs TF-IDF 的 −0.309)。
- 部署结论(当前数据):有 ≥8 条标注/类 时,选 encoder + 线性头——F1、校准、鲁棒性、延迟全面占优;prompt decoder 仅在零标注预算时值得考虑。

曲线图见 [results/learning_curves.png](results/learning_curves.png)(克隆仓库后生成;结果 JSON 已含在仓库)。

## 复现

```bash
pip install scikit-learn sentence-transformers transformers datasets torch
export HF_ENDPOINT=https://hf-mirror.com   # 如 huggingface.co 不可达
python code/load_data.py                   # 下载 AG News/Yahoo 并构建固定池
python code/run_curves.py --arms tfidf,encoder          # 分类器臂(CPU 可跑)
python code/arm_decoder.py                              # decoder 臂(GPU/CPU)
python code/analyze07.py                                # 表格 + 交叉点 + 出图
```

协议细节见 [experiment-plan.md](experiment-plan.md)。

## 待做

- [ ] SetFit 式对比训练对照(真实少样本微调)
- [ ] k 扩到 {4, 16, 64, 256} 补密曲线
- [ ] 量化 decoder 的延迟–成本–F1 三角
