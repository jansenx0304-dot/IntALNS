# IntALNS 会议论文复现材料

本项目对应论文《IntALNS: Interpretable LLM-Agent-Controlled Adaptive Large Neighborhood Search for Cooperative Marine Monitoring Task Allocation》，研究异构海洋监测平台的任务分配与访问顺序优化，对比 Basic 与 IntALNS。

论文唯一入口为 `paper/main.tex`，对应 PDF 为 `paper/main.pdf`。论文为英文，项目使用说明为中文。当前源稿作者栏为空。

## 项目结构

| 路径 | 内容 |
| --- | --- |
| `paper/` | 当前论文、参考文献、SPIE 模板、正文图表及其统计输入 |
| `sar_alloc/` | 问题模型、约束、ALNS 算子、Agent 观察、History、Example、提示与 API 客户端 |
| `experiments/` | 统一实验入口、Basic 与 Agent 适配、独立解校验及统计计算 |
| `configs/` | 固定协议、每规模 Basic 参数、四份实际使用的 Example 库和论文案例定位 |
| `data/test/` | 15 个固定实例、45 个共同起点、实例元数据与生成参数 |
| `results/` | 90 份完整实验记录、90 份运行状态、来源与 SHA-256 校验 |
| `scripts/` | 从归档结果生成当前论文图表，以及干净编译论文 |

`agent` 是 IntALNS 的方法标识。四份 Example 库按规模和搜索状态检索，使用稳定的功能名称标识示例。发布记录仅统一这类标识的命名；目标值、路线、检查点和决策文字保持不变，规范化前的原文件哈希保存在 `results/provenance.json` 的 `source_sha256` 字段中。

## 环境

已验证 Python 3.12.4。建议使用独立 Python 3.12 环境：

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

安装前先激活所建环境：Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`，Linux/macOS 使用 `source .venv/bin/activate`。论文编译另需提供 `pdflatex`、`bibtex` 的 TeX 发行版，包含 amsmath、amsfonts、amssymb、graphicx、booktabs、array、siunitx、algorithm、algpseudocode、placeins、capt-of、tikz、hyperref 等宏包；SPIE 的 class 和 bst 已随包提供。

## 实验协议

两方法使用同一实例、同一起始路线、相同算子与约束，以及相同的 1200 次候选评估预算。算法种子为 0，每 100 次保存一次检查点，IntALNS 同时更新控制；检查点包括 0 和 1200。任务规模为 50、100、300，每规模实例种子为 6800–6804，每实例使用嵌套的 M10、M50、M90 起点，共 45 对条件。

固定目标按优先级损失、未分配任务数、闭合路线转移能耗进行字典序比较。Basic 每规模使用一份参数，来自独立小型调参集（4100、4101，M50）；三个规模均为 removal ratio 0.20、reaction factor 0.10、100 次窗口。IntALNS 的共享 ALNS reaction factor 为 0.20；最终 Example 来自开发数据。运行常量以保留的实现为准，`configs/protocol.json` 记录对应正式设置。

实验使用 `data/test/` 中种子为 6800–6804 的固定实例，完整保留全部 45 对条件、90 份运行结果。原始记录位于 `results/test/seed_0/`，逐记录哈希与对应关系见 `results/provenance.json`，可据此核验并重新统计论文结果。

L1/L2 均值覆盖全部条件；L3 只在配对结果的 L1、L2 分别相等时取均值，表中除以 1000 显示。W/T/L 按完整字典序比较，报告能耗相等容差为相对 1e-10、绝对 1e-7。三个嵌套起点相关，统计中的独立分组单位为底层实例。

## 复现已有论文结果

以下命令从项目根目录执行，不需要 API，不运行搜索：

```bash
python -B scripts/build_evidence.py
python -B scripts/build_paper.py
```

第一条命令校验归档输入，独立重评参考路线、起点和最终解，重新汇总全部 90 份记录，生成 `paper/data/` 中的 CSV/JSON、`paper/tables/` 中的五份 LaTeX 表格输入，以及正文的 `convergence.pdf`、`decision_case.pdf`。方法框架图由 `paper/figures/framework.tikz` 在编译正文时生成。第二条命令执行 LaTeX/BibTeX/LaTeX/LaTeX，生成唯一的 `paper/main.pdf`，自动移除编译中间产物。

若只需重新生成主结果表与数值宏：

```bash
python -B scripts/build_evidence.py --table-only
```

主要核对值：IntALNS 对 Basic 为 30 胜、5 平、10 负；平均优先级损失为 1.82 对 3.62（显示值）。案例定位固定在 T100/6800/M50 的第 7 次决策，属于描述性案例，不是单机制因果实验。

## 重新运行搜索

无需重新生成实例。统一入口读取包内正式配置与起点：

```bash
python -B -m experiments.run --methods basic
python -B -m experiments.run --methods agent
```

新结果默认写入 `outputs/test/seed_0/`，不会覆盖 `results/` 中的归档记录。入口按已完成记录续跑，只对未完成的技术失败重试，最多三次并保留尝试状态。可使用 `--sizes 50`、`--maturities M10` 限定运行范围，或用 `--output` 指定新的输出目录。完整矩阵可能耗时较长，Agent 另需 API 调用。

检查新运行结果时可直接读取输出目录下 `records/` 的 `final_quality`、`validation` 和 `checkpoints`。论文图表脚本专门使用已校验的归档记录，避免新运行输出混入原论文统计。

实例生成实现保留在 `sar_alloc/tools/benchmark_generator.py`。归档复现使用已保存实例；生成器拒绝覆盖已有数据。

## LLM 配置

正式模型名为 `deepseek-v4-flash`，temperature=0、max_tokens=4096、thinking disabled，JSON 输出，单次请求默认超时 60 秒。无效结构按保留实现进行局部修复或最多四次输出修复请求；失败不会作为有效搜索动作执行。

在运行 Agent 前配置 OpenAI 兼容接口的环境变量，密钥和服务地址不随包提供：

```bash
export LLM_API_KEY="由使用者填写"
export LLM_BASE_URL="由使用者填写兼容接口地址"
```

PowerShell 对应 `$env:LLM_API_KEY="..."` 和 `$env:LLM_BASE_URL="..."`。也可使用 `--env-file` 指定项目外的环境文件，变量名见 `.env.example`。默认不读取系统代理；确有需要时设置 `LLM_USE_SYSTEM_PROXY=1`。

外部服务的可用性、模型版本及采样实现会影响重新调用得到的决策，即使 temperature=0 也不保证逐字节重现。论文表格和曲线可从本包保存的原始记录确定性重建。
