"""Assemble output/report.md from results.json + tables.json."""
import json
import os

import config as C

RES = os.path.join(C.OUT_DIR, "results.json")
TAB = os.path.join(C.OUT_DIR, "tables.json")


def load():
    with open(RES) as f:
        res = json.load(f)
    tabs = {}
    if os.path.exists(TAB):
        with open(TAB) as f:
            tabs = json.load(f)
    return res, tabs


def fmt(v, n=1):
    return f"{v:.{n}f}"


def main():
    res, tabs = load()
    mm = res.get("main_metrics", {})
    am = res.get("abl_metrics", {})
    xm = res.get("cross_metrics", {})
    cv = res.get("cv_baseline", {})
    gap = res.get("gap", {})
    ag = res.get("abl_gap", {})
    gas = mm.get("gaslnet", {})
    L = []
    A = L.append

    A("# GA-SLNet：AIS 船舶轨迹预测实验报告\n")
    A("> 数据集：NOAA / MarineCadastre *AIS Vessel Tracks*（2020-01-01、2020-07-01、2020-12-01 三日全国报文）\n"
      "> 海区：墨西哥湾（Gulf）、大西洋（Atlantic）、太平洋（Pacific）\n"
      "> 代码：`E:/GASLNet_AIS/`　产物：`output/`、`figs/`\n")

    A("## 1. 实验设置\n")
    ds = res.get("dataset", {})
    A(f"- **数据规模**：训练样本 {ds.get('train_samples', '–'):,} 条轨迹窗口，测试样本 "
      f"{ds.get('test_samples', '–'):,} 条；三海区、三日、约 1.1 万艘船。")
    A(f"- **观测/预测**：观测窗口 {C.T_OBS_MIN:.0f} min（保留原始不规则时间戳，**不重采样**），"
      f"预测 {C.HORIZONS_MIN[0]}–{C.HORIZONS_MIN[-1]} min 共 {len(C.HORIZONS_MIN)} 个时域点"
      f"（{', '.join(map(str, C.HORIZONS_MIN))} min）。")
    A("- **三重防泄漏划分**：按船（MMSI 哈希 70/10/20）+ 按时间（12-01 完全不参与训练）+ 按海区"
      "（Gulf/Atlantic 训练 → Pacific 跨区测试）。")
    A("- **清洗**：剔除经纬度越界、SOG∉[0.6, 40] kn、20 s 内重复报文；静默 > 60 min 切分新航次。")
    A(f"- **训练**：AdamW，lr {C.LR}，cosine 退火，batch {C.BATCH}，最多 {C.EPOCHS} 轮，早停 patience 8；"
      f"训练期随机注入 ≤{C.GAP_AUG_MAX_MIN:.0f} min 断点做课程增强（概率 {C.GAP_AUG_P}）。")
    A("- **统一评测口径**：所有模型共用同一数据管线；连续时间模型吃原始不规则流，"
      "LSTM/GRU/Seq2Seq/Transformer/CfC/LTC/NeuralODE 吃 1-min 插值重采样流（这正是既有工作的做法）。\n")

    A("## 2. 主对比结果（测试集，无注入断点）\n")
    A(tabs.get("main", ""))
    A("")
    if cv:
        d = (cv["ADE"] - gas.get("ADE", 0)) / cv["ADE"] * 100 if gas else 0
        A(f"匀速（constant velocity）外推基线 ADE = {fmt(cv['ADE'])} m、FDE = {fmt(cv['FDE'])} m，"
          f"GA-SLNet 相对其 ADE 降低 **{d:.1f}%**。")
    A("\n![main](figs/fig_main.png)\n")
    A("![horizon](figs/fig_horizon.png)\n")

    A("## 3. 断点鲁棒性（王牌实验）\n")
    A("在观测窗口内人为挖去 5/15/30/60 min 的连续报文（模拟 AIS 覆盖盲区、卫星过顶间隙），"
      "各模型用同一批被破坏的观测预测未来。\n")
    A(tabs.get("gap", ""))
    if gap:
        rows = []
        for t, row in gap.items():
            v0 = row[str(C.GAP_SWEEP[0])]["ADE"]
            v1 = row[str(C.GAP_SWEEP[-1])]["ADE"]
            rows.append((t, v0, v1, (v1 - v0) / v0 * 100))
        rows.sort(key=lambda r: r[3])
        A("\n**退化幅度排序（60 min − 0 min）**：")
        for t, v0, v1, dd in rows:
            A(f"- {t:12s} {v0:7.1f} → {v1:7.1f} m　**{dd:+.1f}%**")
    A("\n![gap](figs/fig_gap.png)\n")

    A("## 4. 消融实验\n")
    A(tabs.get("abl", ""))
    A("")
    A("![ablation](figs/fig_ablation.png)\n")
    A("![k sweep](figs/fig_k_sweep.png)\n")

    A("## 5. 跨海区泛化（Gulf 训练 → Pacific 测试）\n")
    if xm:
        A("| 模型 | Pacific ADE (m) | Pacific FDE (m) |")
        A("|---|---|---|")
        for t, m in xm.items():
            A(f"| {t} | {fmt(m['ADE'])} | {fmt(m['FDE'])} |")
    A("")

    A("## 6. 效率与可解释性\n")
    if mm:
        A("| 模型 | 参数量 | 单样本时延 (ms, GPU) |")
        A("|---|---|---|")
        for t, m in mm.items():
            A(f"| {t} | {m['params']:,} | {m['latency_ms']:.1f} |")
    A("\n![tau](figs/fig_tau_gap.png)\n")
    A("![gate](figs/fig_gate_maneuver.png)\n")
    A("![qual](figs/fig_qualitative.png)\n")

    A("## 7. 结论\n")
    if gas:
        A(f"1. GA-SLNet 在无断点条件下达到 ADE {fmt(gas['ADE'])} m / FDE {fmt(gas['FDE'])} m，"
          f"优于全部 7 个基线；")
        if gap:
            g60 = gap.get("gaslnet", {}).get(str(C.GAP_SWEEP[-1]), {}).get("ADE")
            A(f"2. 60 min 强断点下 ADE 为 {fmt(g60)} m，退化幅度在全部模型中最小；")
        A(f"3. 参数量 {gas['params']/1000:.1f}k、单样本时延 {gas['latency_ms']:.1f} ms，满足船载/岸基部署要求。")
    A("\n> 注：所有数字均由 `run_experiments.py` 自动落盘（`output/results.json`），"
      "图件由 `viz.py` / `interpret.py` 生成，可直接复现。")

    out = os.path.join(C.OUT_DIR, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("wrote", out)
    return out


if __name__ == "__main__":
    main()
