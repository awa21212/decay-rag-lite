# -*- coding: utf-8 -*-
"""Four corpora: does mixing them hurt? Where does it break?

Runs the same labelled questions under three conditions:

1. ``aligned``  — only the corpus the question belongs to (oracle routing)
2. ``mixed``    — all four corpora in one pool (what preflight.py actually does)
3. ``sweep``    — mixed pool padded with filler chunks, to find the size at
                  which hit@1 falls off a cliff

Writes ``docs/four_corpus_bench.png`` and prints a markdown table.
"""
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import ask  # noqa: E402
import corpora  # noqa: E402

NAMES = ("code", "finance", "task", "life")

CASES = [
    ("docker 里 apt 报 502 怎么办", "code", "01-docker-apt-502.md"),
    ("C++ 的文件时间戳和别的语言差几个小时", "code", "02-cpp-mtime-epoch.md"),
    ("fitness 上不去，换手太高", "finance", "01-fitness-turnover.md"),
    ("样本外那条快腿为什么变成噪声了", "finance", "fitness_2026-11-10.md"),
    ("接任务之前先看什么", "task", "01-preflight.md"),
    ("多语言移植验证做完了吗", "task", "status_2026-11-16.md"),
    ("旧阵列拆了之后什么感觉", "life", "journal_2026-11-14.md"),
    ("备份脚本写了没有", "life", "journal_2026-11-14.md"),
]

FILLER = (
    "今天把仓库里的旧分支清理了一遍，顺手把提交信息统一成约定格式，"
    "下午重新跑了一次全量测试，两个用例因为网络超时失败，重跑就过了。"
    "晚上整理了待办清单，把下周要做的三件事按优先级排了序。"
)


def load_all():
    pools = {}
    for n in NAMES:
        mat, meta = ask.load_corpus(n)
        pools[n] = (mat, meta)
    return pools


def score_pool(q, mat, meta):
    vec = np.array(ask.embed_one(q), dtype="float32")
    cos = np.clip(mat @ vec, 0, 1)
    qb = ask.bigrams(q)
    lex = np.array(
        [len(qb & ask.bigrams(m["text"])) / (len(qb) + 1e-9) for m in meta],
        dtype="float32",
    )
    if lex.max() > 0:
        lex = lex / lex.max()
    sim = 0.72 * cos + 0.28 * lex
    now = time.time()
    ts = np.array([ask.ts_of(m) for m in meta], dtype="float64")
    age = np.maximum(0.0, (now - ts) / 86400.0)
    decay = 0.5 ** (age / 30.0)
    return 0.75 * sim + 0.25 * decay


def hit_at_1(final, meta, target):
    order = np.argsort(-final)[:3]
    names = [os.path.basename(meta[i]["path"]) for i in order]
    return names[0] == target, names


def decoy_chunks(per_case, seed=7):
    """Near-miss documents: they repeat the question's words but are not it."""
    rnd = random.Random(seed)
    out = []
    for q, _, _ in CASES:
        for i in range(per_case):
            body = q + "。" + "".join(rnd.choice(FILLER) for _ in range(2))
            out.append({"path": "decoy/%s_%02d.md" % (abs(hash(q)) % 10000, i), "text": body})
    return out


def main():
    ask.clear_proxy()
    pools = load_all()
    mats = np.vstack([pools[n][0] for n in NAMES])
    metas = [m for n in NAMES for m in pools[n][1]]

    rows = []
    for q, corpus, target in CASES:
        single = score_pool(q, *pools[corpus])
        mixed = score_pool(q, mats, metas)
        ok_s, names_s = hit_at_1(single, pools[corpus][1], target)
        ok_m, names_m = hit_at_1(mixed, metas, target)
        rows.append({"q": q, "corpus": corpus, "target": target,
                     "single": ok_s, "mixed": ok_m,
                     "single_top": names_s[0], "mixed_top": names_m[0]})

    # per-corpus accuracy when every question is asked against that corpus only
    cross = {}
    for n in NAMES:
        hits = 0
        for q, corpus, target in CASES:
            final = score_pool(q, *pools[n])
            ok, _ = hit_at_1(final, pools[n][1], target)
            hits += 1 if ok else 0
        cross[n] = hits / len(CASES)

    aligned = sum(r["single"] for r in rows) / len(rows)
    mixed = sum(r["mixed"] for r in rows) / len(rows)

    # size sweep: pad the mixed pool with filler chunks
    sweep = []
    for per in (0, 1, 2, 4, 8, 16):
        docs = decoy_chunks(per) if per else []
        if docs:
            fmat = np.array([ask.embed_one(d["text"]) for d in docs], dtype="float32")
            fmat = fmat / np.linalg.norm(fmat, axis=1, keepdims=True)
            pool_mat = np.vstack([mats, fmat])
            pool_meta = metas + docs
        else:
            pool_mat, pool_meta = mats, metas
        hits = 0
        for q, corpus, target in CASES:
            final = score_pool(q, pool_mat, pool_meta)
            ok, _ = hit_at_1(final, pool_meta, target)
            hits += 1 if ok else 0
        sweep.append({"blocks": len(pool_meta), "acc": hits / len(CASES)})

    out = {"aligned": aligned, "mixed": mixed, "cross": cross, "sweep": sweep,
           "cases": rows}
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("| 条件 | hit@1 |\n|---|---|")
    print("| 只查对口库（oracle） | %.2f |" % aligned)
    print("| 混合四库 | %.2f |" % mixed)
    for n in NAMES:
        print("| 只用 %s 库答全部题 | %.2f |" % (n, cross[n]))
    print("\n规模扫描：")
    for s in sweep:
        print("  %4d 块 → %.2f" % (s["blocks"], s["acc"]))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
            if any(f in x.name for x in font_manager.fontManager.ttflist):
                plt.rcParams["font.sans-serif"] = [f]
                break
        plt.rcParams["axes.unicode_minus"] = False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})

        labels = ["对口单库", "混合四库"] + ["只用 %s" % n for n in NAMES]
        vals = [aligned, mixed] + [cross[n] for n in NAMES]
        colors = ["#2f6f9f", "#c0504d"] + ["#9aa5b1"] * 4
        bars = ax1.bar(labels, vals, color=colors)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, "%.2f" % v,
                     ha="center", fontsize=9)
        ax1.set_ylim(0, 1.12)
        ax1.set_ylabel("hit@1（8 道题）")
        ax1.set_title("单独使用 vs 混合使用（同一批题）")
        ax1.grid(axis="y", alpha=0.25)

        xs = [s["blocks"] for s in sweep]
        ys = [s["acc"] for s in sweep]
        ax2.plot(xs, ys, marker="o", color="#2f6f9f")
        ax2.axhline(0.8, ls="--", lw=1, color="#c0504d")
        ax2.text(xs[0], 0.82, "0.80 警戒线", color="#c0504d", fontsize=8)
        for x, y in zip(xs, ys):
            ax2.annotate("%.2f" % y, (x, y), textcoords="offset points", xytext=(0, 7),
                         ha="center", fontsize=8)
        ax2.set_xlabel("语料总块数（对数）")
        ax2.set_ylabel("hit@1")
        ax2.set_xscale("log", base=2)
        ax2.set_ylim(0, 1.12)
        ax2.set_title("边界：语料变大后混合检索的衰减")
        ax2.grid(alpha=0.25)

        fig.suptitle("四库 RAG：单库 vs 混合，以及规模边界", fontsize=13)
        fig.tight_layout()
        png = os.path.join(ROOT, "docs", "four_corpus_bench.png")
        os.makedirs(os.path.dirname(png), exist_ok=True)
        fig.savefig(png, dpi=150)
        print("\n图已保存：", png)
    except Exception as exc:  # matplotlib 缺失时不影响表格
        print("（跳过画图：%s）" % exc)


if __name__ == "__main__":
    main()
