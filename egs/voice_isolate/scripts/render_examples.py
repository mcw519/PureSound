"""Build the local v8-vs-v11 listening page from the field benchmark clips.

For each selected clip: run both checkpoints at ``dry_blend 0.9`` on CPU (the
same call ``eval_realcase.py`` makes, so the numbers below reproduce the
``benchmarks/full_gate/`` records), write 16 kHz mono mp3 for mix / v8 / v11 /
QVF2.2 reference, render a stacked spectrogram whose top ruler carries the
hand-labelled keep/suppress spans, measure every span on its own, and emit an
``index.html`` that plays the four systems from a shared clock.

The output tree holds internal field audio, so it lives under the ignored
``data_report/`` and is never committed or uploaded.

    uv run python egs/voice_isolate/scripts/render_examples.py \
        egs/voice_isolate/data_report/listening/v8_vs_v11
"""
from __future__ import annotations

import html
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from puresound.audio.io import AudioIO  # noqa: E402
from eval_realcase import (  # noqa: E402
    KEEP_VIOLATION_DB,
    SUPPRESS_FAIL_DB,
    SUPPRESS_PARTIAL_DB,
    load_model,
    span_dbfs,
)

EGS = REPO_ROOT / "egs/voice_isolate"
CASES_DIR = EGS / "data_report/field_cases/test_vector_cases"
CONFIG = EGS / "config/infer_dpcrn.yaml"
CKPTS = {"v8": EGS / "pretrained_ckpt/dpcrn_v8.ckpt",
         "v11": EGS / "pretrained_ckpt/backup/dpcrn_v11_ep19.ckpt"}
BLEND = 0.9
SR = 16000
MIN_HEADROOM = 6.0

LABELS = {"mix": "未處理", "v8": "dpcrn_v8", "v11": "dpcrn_v11_ep19", "qvf22": "QVF2.2 參考"}

# clip -> (excerpt window in seconds or None, headline, reading)
CASES = {
    "qvf_scenario3_session": (
        None,
        "跨擷取鏈牆本身，旁邊放著商用參考",
        "我們兩版在 SUPPRESS 段幾乎沒推（−0.8 dB ＝ passthrough），QVF2.2 把同一段壓到底噪以下 19 dB——"
        "頻譜圖最下面那一格整片黑，就是這道牆。這一格也解釋了為什麼四欄都得報："
        "residual 9.6 dB 其實低於 12 dB 的 PARTIAL 門檻，光看殘留會誤判成「還可以」，"
        "判 SUPPRESS-FAIL 的是 reduc——因為這支錄音的 headroom 本來就只有 10.4 dB。",
    ),
    "180d_session": (
        (24.0, 54.0),
        "held-out 朝向：v11 的壓制深 4.8 dB",
        "180D 是程式強制 held-out 的第三個裝置朝向，沒有任何讀出在它上面擬合過。"
        "整段彙總是 v11 把殘留壓掉 4.8 dB、殘留與使用者聲音的距離拉開到 22.6 dB，兩版都仍是 SUPPRESS-PARTIAL。"
        "但逐段表把彙總藏起來的事情攤開了：這 4.8 dB 幾乎全部來自 SUPPRESS #1（13.2–25.8 s，"
        "v8 只推 −2.59 dB ＝ 接近放行，v11 −9.08 dB）；後三段 v8 本來就做到 −14～−18 dB，v11 沒有系統性更好"
        "（#3 甚至退 3.8 dB）。#1 是長 KEEP 開場之後的第一個遠場輪次——"
        "跟『證據累積』那組探針的方向一致：模型在第一個遠場輪次上最弱。"
        "KEEP 側兩版都守住（逐段 −0.01～−1.49，唯一例外是 v11 在 KEEP #4 掉到 −1.21）。",
    ),
    "90d_far2": (
        None,
        "v11 唯一收復的遠場段：從 passthrough 到 −9 dB",
        "v8 在這一段幾乎完全放行（reduc −0.17 dB、殘留 31.1 dB），v11 推了 9 dB。"
        "這是 v11 把冷啟動遠場判決從 ok 2 變成 ok 3 的那一格——也是整份對照裡差異最容易用耳朵聽出來的一段。",
    ),
    "qvf_keep_in_touch_near1": (
        None,
        "v11 的代價：QVF 鏈上把使用者削掉 19 dB",
        "v8 只是剛越過 −3 dB 門檻，v11 把使用者削掉 19 dB——機器基本上聽不到人在講話。"
        "同一支錄音的 double-talk 段方向相反（v8 −11.79 → v11 −9.34，兩版都違規），"
        "所以這不是單一 span 的抽樣意外，而是整條 QVF 鏈上的 keep 行為在移動。",
    ),
    "qvf_gym_near1": (
        None,
        "v11 唯一新增的 keep 違規",
        "v8 通過、v11 不通過。兩版所有 keep 違規（除了 0D sentinel）都落在 QVF 鏈上——"
        "裝置錄音（90d／180d／270d）是零。這就是跨鏈牆在成績單上的另一面。",
    ),
    "qvf_price_far2": (
        None,
        "同一條 QVF 鏈，壓制側卻做得到 −20 dB",
        "殘留是負的＝輸出已經沉到房間底噪之下，旁人真的消失了；v11 再往下 3.4 dB。"
        "放這一格是因為它反駁一個容易得出的錯誤結論：QVF 鏈不是「整條都壞」——"
        "壞的是 keep 側（把使用者判成遠場），壓制側在同一條鏈上能做到 −20 dB。",
    ),
    "0d_near1": (
        None,
        "sentinel：已出貨預設在高底噪擷取上刪使用者",
        "0D 是比其他錄音吵 25 dB 的擷取，只有 14.2 dB 的 SNR headroom。"
        "同一個 v8 在乾淨擷取上削 0.05–0.81 dB，在這裡削 4.05 dB；v11 一樣違規（−4.23）。"
        "這是靠把測試集加寬找出來的、發生在已出貨預設上的 keep 失敗，不是任何模型改動造成的——"
        "v2 田野集裡沒有這種條件的錄音，所以任何 v2 成績單都不可能顯示它。",
    ),
}

PANEL_BG = "#0f1519"
FG = "#c9d6de"
KEEP_C = "#4fb783"
SUPP_C = "#e08a76"


def enhance(model, wav: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return (model(wav, dry_blend=BLEND, spec_floor=0.0, presence_gate=None)
                .detach().cpu().view(1, -1).clamp(-1.0, 1.0))


def write_audio(wav: torch.Tensor, stem: Path) -> int:
    """16 kHz mono mp3 for the page; the wav stays so pictures can be redrawn."""
    sf.write(stem.with_suffix(".wav"), wav.view(-1).numpy().astype(np.float32), SR, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(stem.with_suffix(".wav")),
                    "-ar", "16000", "-ac", "1", "-b:a", "48k", str(stem.with_suffix(".mp3"))],
                   check=True)
    return stem.with_suffix(".mp3").stat().st_size


def spec_db(x: np.ndarray) -> np.ndarray:
    n_fft, hop = 512, 128
    win = np.hanning(n_fft).astype(np.float32)
    frames = 1 + max(0, (len(x) - n_fft)) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    return 20.0 * np.log10(np.abs(np.fft.rfft(x[idx] * win, axis=-1)).T + 1e-7)


def render_png(panels: list[tuple[str, np.ndarray]], spans: dict, dur: float, path: Path) -> None:
    n = len(panels)
    fig, axes = plt.subplots(n + 1, 1, figsize=(11.0, 0.62 + 1.55 * n), dpi=104,
                             gridspec_kw={"height_ratios": [0.42] + [1.0] * n, "hspace": 0.14})
    fig.patch.set_facecolor(PANEL_BG)

    ruler = axes[0]
    ruler.set_facecolor(PANEL_BG)
    ruler.set_xlim(0, dur)
    ruler.set_ylim(0, 1)
    for kind, colour, label in (("keep", KEEP_C, "KEEP"), ("suppress", SUPP_C, "SUPPRESS")):
        for a, b in spans.get(kind, []):
            if b <= 0 or a >= dur:
                continue
            a, b = max(a, 0.0), min(b, dur)
            ruler.add_patch(plt.Rectangle((a, 0.30), b - a, 0.40, color=colour, alpha=0.85, lw=0))
            if b - a > dur * 0.055:
                ruler.text((a + b) / 2, 0.50, label, ha="center", va="center", color=PANEL_BG,
                           fontsize=6.6, fontweight="bold", family="monospace")
    for spine in ruler.spines.values():
        spine.set_visible(False)
    ruler.set_xticks([])
    ruler.set_yticks([])
    ruler.text(0, 1.05, "hand-labelled spans", color=FG, fontsize=7.0, family="monospace", va="bottom")

    bounds = sorted({round(t, 3) for kind in ("keep", "suppress")
                     for span in spans.get(kind, []) for t in span if 0 < t < dur})

    vmax = None
    for ax, (label, x) in zip(axes[1:], panels):
        S = spec_db(x)
        if vmax is None:
            vmax = float(np.percentile(S, 99.7))
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                  vmin=vmax - 72, vmax=vmax, extent=[0, dur, 0, SR / 2000])
        ax.set_facecolor(PANEL_BG)
        for t in bounds:
            ax.axvline(t, color="#e9f2f7", lw=0.7, ls=(0, (3, 3)), alpha=0.55)
        ax.set_ylabel("kHz", color=FG, fontsize=7.2, labelpad=2)
        ax.tick_params(colors=FG, labelsize=6.6, length=2, width=0.5)
        ax.set_yticks([0, 4, 8])
        for spine in ax.spines.values():
            spine.set_color("#2b373f")
        ax.text(0.006, 0.92, label, transform=ax.transAxes, color="#f2f8fb", fontsize=8.2,
                fontweight="bold", family="monospace", va="top",
                bbox=dict(facecolor=PANEL_BG, edgecolor="none", alpha=0.72, pad=1.6))
        if ax is not axes[-1]:
            ax.set_xticklabels([])
    axes[-1].set_xlabel("seconds", color=FG, fontsize=7.2)
    fig.savefig(path, facecolor=PANEL_BG, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def verdict(keep: float | None, reduc: float | None, residual: float | None,
            headroom: float | None) -> str:
    flags = []
    if keep is not None and keep < KEEP_VIOLATION_DB:
        flags.append("KEEP-VIOLATION")
    if reduc is not None:
        if headroom is not None and headroom < MIN_HEADROOM:
            flags.append("NOT-SCORABLE")
        elif reduc > SUPPRESS_FAIL_DB:
            flags.append("SUPPRESS-FAIL")
        elif residual is not None and residual > SUPPRESS_PARTIAL_DB:
            flags.append("SUPPRESS-PARTIAL")
    return ",".join(flags) if flags else "ok"


def r2(v: float | None) -> float | None:
    return None if v is None or v != v else round(float(v), 2)


def measure_spans(wav: torch.Tensor, spans: list, n: int) -> float:
    return span_dbfs(wav, spans, SR, n) if spans else float("nan")


# --------------------------------------------------------------------------- page


CSS = """
:root{--bg:#0b1014;--card:#131b21;--line:#26333b;--ink:#e6eef3;--dim:#93a4b0;
 --accent:#55bcd4;--pass:#63c294;--fail:#e08a76;--hold:#d5ac52;
 --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
 --sans:"Noto Sans TC","PingFang TC",system-ui,sans-serif;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-weight:300;
 line-height:1.8;font-size:16px}
.wrap{max-width:66rem;margin:0 auto;padding:3rem 1.5rem 5rem;display:flex;flex-direction:column;gap:3rem}
h1{font-size:clamp(1.6rem,4vw,2.3rem);margin:0;font-weight:700;line-height:1.3}
h2{font-size:1.15rem;margin:0;font-weight:500;line-height:1.5}
.eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;
 color:var(--accent)}
p{margin:0;max-width:52rem}
p+p{margin-top:.8rem}
.dim{color:var(--dim)}
.m,code{font-family:var(--mono);font-size:.86em}
strong{font-weight:500;color:#fff}
.howto{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);
 border-radius:2px;padding:1.2rem 1.35rem;display:flex;flex-direction:column;gap:.7rem}
.howto p{font-size:.92rem;color:var(--dim)}
.case{background:var(--card);border:1px solid var(--line);border-radius:3px;
 padding:1.4rem 1.45rem;display:flex;flex-direction:column;gap:1.1rem}
.case-head{display:flex;flex-direction:column;gap:.45rem}
.cid{font-family:var(--mono);font-size:.85rem;color:var(--accent);font-weight:500}
.meta{font-family:var(--mono);font-size:.72rem;color:var(--dim);display:flex;flex-wrap:wrap;
 gap:.3rem 1.1rem}
.flag{display:inline-block;font-family:var(--mono);font-size:.62rem;font-weight:600;
 letter-spacing:.1em;text-transform:uppercase;padding:.18em .5em;border-radius:2px;
 background:#1b3239;color:var(--accent)}
.players{display:flex;flex-wrap:wrap;gap:.55rem}
.sysbtn{font-family:var(--mono);font-size:.78rem;padding:.5rem .9rem;border-radius:2px;
 border:1px solid var(--line);background:#1a2429;color:var(--ink);cursor:pointer}
.sysbtn:hover{border-color:var(--accent)}
.sysbtn.on{background:var(--accent);color:#08161b;border-color:var(--accent);font-weight:600}
.sysbtn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.clock{font-family:var(--mono);font-size:.78rem;color:var(--dim);align-self:center;
 font-variant-numeric:tabular-nums}
.chips{display:flex;flex-wrap:wrap;gap:.4rem}
.chip{font-family:var(--mono);font-size:.7rem;padding:.3rem .6rem;border-radius:2px;cursor:pointer;
 border:1px solid transparent;font-variant-numeric:tabular-nums}
.chip.keep{background:#163024;color:var(--pass)}
.chip.supp{background:#34201b;color:var(--fail)}
.chip:hover{border-color:currentColor}
.chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
img.spec{width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:2px}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:2px}
table{width:100%;border-collapse:collapse;font-size:.85rem;min-width:34rem}
th{text-align:left;font-family:var(--mono);font-size:.64rem;letter-spacing:.11em;
 text-transform:uppercase;color:var(--dim);padding:.6rem .8rem;border-bottom:1px solid var(--line);
 white-space:nowrap;font-weight:600}
td{padding:.55rem .8rem;border-bottom:1px solid #1d272e;color:var(--dim);white-space:nowrap;
 font-family:var(--mono);font-variant-numeric:tabular-nums}
td.name{font-family:var(--sans);color:var(--ink)}
tbody tr:last-child td{border-bottom:none}
.v-ok{color:var(--pass)}
.v-bad{color:var(--fail)}
.v-part{color:var(--hold)}
.read{font-size:.92rem;color:var(--dim);line-height:1.8}
footer{font-family:var(--mono);font-size:.72rem;color:var(--dim);border-top:1px solid var(--line);
 padding-top:1.3rem;line-height:1.9}
audio{display:none}
"""

JS = """
function initCase(id, tags){
  const el = t => document.getElementById(id+'-a-'+t);
  const btns = tags.map(t => document.getElementById(id+'-b-'+t));
  const clock = document.getElementById(id+'-clock');
  let cur = tags[0], pos = 0;
  const mark = () => btns.forEach((b,i) => b.classList.toggle('on', tags[i]===cur));
  const show = () => { clock.textContent = pos.toFixed(2)+' s'; };
  tags.forEach(t => {
    const a = el(t);
    a.addEventListener('timeupdate', () => { if(!a.paused){ pos = a.currentTime; show(); } });
    a.addEventListener('ended', () => { pos = 0; show(); });
  });
  function play(t){
    tags.forEach(o => { if(o!==t){ const a=el(o); a.pause(); } });
    cur = t; mark();
    const a = el(t); a.currentTime = pos;
    if(a.paused){ a.play(); } else { a.pause(); }
  }
  function seek(s){
    pos = s; show();
    const a = el(cur);
    a.currentTime = s;
    tags.forEach(o => { if(o!==cur) el(o).pause(); });
    a.play();
  }
  tags.forEach((t,i) => btns[i].addEventListener('click', () => play(t)));
  document.querySelectorAll('[data-seek][data-case="'+id+'"]').forEach(c =>
    c.addEventListener('click', () => seek(parseFloat(c.dataset.seek))));
  mark(); show();
}
"""


def vclass(v: str) -> str:
    if v == "ok":
        return "v-ok"
    return "v-part" if "PARTIAL" in v else "v-bad"


def fmt(v: float | None, unit: str = "") -> str:
    return "—" if v is None else f"{v:+.2f}{unit}" if unit == "" else f"{v:.2f}{unit}"


def build_page(report: dict, out: Path) -> None:
    parts = [
        '<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>v8 vs v11 聽測頁 — voice_isolate</title>",
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;500;600&family=Noto+Sans+TC:wght@300;400;500;700&display=swap">',
        f"<style>{CSS}</style></head><body><div class='wrap'>",
        "<header style='display:flex;flex-direction:column;gap:1rem'>",
        "<div class='eyebrow'>本機聽測頁 · 不上傳 · data_report/ (gitignored)</div>",
        "<h1>dpcrn_v8 vs dpcrn_v11_ep19 — 逐段聽測</h1>",
        "<p class='dim'>兩版都用 <span class='m'>dry_blend 0.9</span>、CPU、"
        "與 <span class='m'>eval_realcase.py</span> 完全相同的推論呼叫。"
        "頁面上每個數字都是從你正在聽的那段音訊重算的，"
        "已驗證與 <span class='m'>benchmarks/full_gate/</span> 的紀錄相符。</p>",
        "</header>",
        "<div class='howto'><h2>怎麼用</h2>",
        "<p><strong>切換系統</strong>：按 未處理 / v8 / v11 / QVF2.2 會從<em>同一個時間點</em>接著播，"
        "所以 A/B 不用重新找位置；再按一次同一顆是暫停。</p>",
        "<p><strong>跳到要檢視的段落</strong>：按 <span class='m'>KEEP</span>（綠）或 "
        "<span class='m'>SUPPRESS</span>（橘）標籤就跳到那一段的起點。"
        "標籤的時間就是 <span class='m'>windows.json</span> 的手工標記，"
        "頻譜圖最上方的 ruler 與貫穿虛線畫的是同一組邊界。</p>",
        "<p><strong>判決門檻</strong>："
        "<span class='m'>reduc &gt; −6 dB</span> = SUPPRESS-FAIL（沒推）、"
        "<span class='m'>residual &gt; 12 dB</span> = SUPPRESS-PARTIAL（推了但旁人仍在房間音之上）、"
        "<span class='m'>keep &lt; −3 dB</span> = KEEP-VIOLATION（對使用者變聾）。"
        "<span class='m'>residual</span> ＝ 輸出高於該支錄音房間底噪多少 dB，"
        "<span class='m'>sir_out</span> ＝ 殘留比使用者自己的聲音低多少。</p>",
        "</div>",
    ]

    for clip, info in report.items():
        tags = list(info["metrics"].keys())
        flags = []
        if info["held_out"]:
            flags.append("held out")
        if info["sentinel"]:
            flags.append("sentinel")
        if "qvf22" in tags:
            flags.append("has reference")
        parts += [
            "<section class='case'>",
            "<div class='case-head'>",
            f"<div class='cid'>{html.escape(clip)}"
            + "".join(f" <span class='flag'>{f}</span>" for f in flags) + "</div>",
            f"<h2>{html.escape(info['headline'])}</h2>",
            f"<p class='dim' style='font-size:.86rem'>{html.escape(info['role'])}</p>",
            "<div class='meta'>"
            + f"<span>floor {info['floor_dbfs']:.2f} dBFS</span>"
            + (f"<span>near_ref {info['near_ref_dbfs']:.2f} dBFS</span>"
               if info["near_ref_dbfs"] == info["near_ref_dbfs"] else "")
            + (f"<span>headroom {info['headroom_db']:.2f} dB</span>"
               if info["headroom_db"] is not None else "")
            + f"<span>clip {info['duration_s']:.2f} s</span>"
            + (f"<span>本頁片段 {info['excerpt_s'][0]:.1f}–{info['excerpt_s'][1]:.1f} s</span>"
               if info["excerpt_s"] != [0.0, info["duration_s"]] else "")
            + "</div>",
            "</div>",
        ]

        for t in tags:
            parts.append(f"<audio id='{clip}-a-{t}' src='{clip}/{t}.mp3' preload='metadata'></audio>")
        parts.append("<div class='players'>")
        for t in tags:
            parts.append(f"<button class='sysbtn' id='{clip}-b-{t}' type='button'>{LABELS[t]}</button>")
        parts.append(f"<span class='clock' id='{clip}-clock'>0.00 s</span></div>")

        chips = []
        for kind, cls, label in (("keep", "keep", "KEEP"), ("suppress", "supp", "SUPPRESS")):
            for a, b in info["spans_excerpt"].get(kind, []):
                chips.append(
                    f"<button class='chip {cls}' type='button' data-case='{clip}' "
                    f"data-seek='{a:.3f}'>{label} {a:.2f}–{b:.2f} s</button>")
        if chips:
            parts.append("<div class='chips'>" + "".join(chips) + "</div>")

        parts.append(f"<img class='spec' src='{clip}/spec.png' "
                     f"alt='{html.escape(clip)} 頻譜對照：mix / v8 / v11'>")

        has_keep = any("keep" in m for m in info["metrics"].values())
        has_supp = any("reduc" in m for m in info["metrics"].values())
        head = ["系統"]
        if has_keep:
            head += ["keep dB", "keep out dBFS"]
        if has_supp:
            head += ["reduc dB", "out dBFS", "residual dB", "sir_out dB"]
        head.append("判決")
        parts.append("<div class='scroll'><table><thead><tr>"
                     + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>")
        for t in tags:
            m = info["metrics"][t]
            cells = [f"<td class='name'>{LABELS[t]}</td>"]
            if has_keep:
                cells += [f"<td>{fmt(m.get('keep'))}</td>", f"<td>{fmt(m.get('keep_out'),' ')}</td>"]
            if has_supp:
                cells += [f"<td>{fmt(m.get('reduc'))}</td>", f"<td>{fmt(m.get('supp_out'),' ')}</td>",
                          f"<td>{fmt(m.get('residual'))}</td>", f"<td>{fmt(m.get('sir_out'),' ')}</td>"]
            v = m["verdict"]
            cells.append(f"<td class='{vclass(v)}'>{v}</td>")
            parts.append("<tr>" + "".join(cells) + "</tr>")
        parts.append("</tbody></table></div>")

        per_span = info.get("per_span") or []
        if len(per_span) > 1:
            sys_cols = [t for t in tags if t != "mix"]
            parts.append("<div class='scroll'><table><thead><tr><th>段落</th><th>時間</th>"
                         + "".join(f"<th>{LABELS[t]}</th>" for t in sys_cols)
                         + "</tr></thead><tbody>")
            for row in per_span:
                cells = [f"<td class='name'>{row['kind'].upper()} #{row['i']}</td>",
                         f"<td>{row['t0']:.2f}–{row['t1']:.2f}</td>"]
                for t in sys_cols:
                    val = row["by_system"].get(t)
                    cells.append(f"<td>{fmt(val)}</td>")
                parts.append("<tr>" + "".join(cells) + "</tr>")
            parts.append("</tbody></table></div>")
            parts.append("<p class='dim' style='font-size:.8rem'>"
                         "逐段欄位：KEEP 段是 keep dB（越接近 0 越好），"
                         "SUPPRESS 段是 reduc dB（越負越好）。時間是整支 clip 的時間軸。</p>")

        parts.append(f"<p class='read'>{html.escape(info['reading'])}</p>")
        parts.append("</section>")

    parts += [
        "<footer>",
        "生成：<span class='m'>egs/voice_isolate/scripts/render_examples.py</span> · "
        "標記來源：<span class='m'>benchmarks/field_test_vector/spans/</span> → "
        "<span class='m'>windows.json</span> · "
        "數值定義：<span class='m'>benchmarks/field_test_vector/RESULTS.md</span><br>"
        "音檔為內部田野錄音的衍生物，只存在本機 <span class='m'>data_report/</span>；不進版控、不上傳任何託管服務。",
        "</footer>",
        f"<script>{JS}</script><script>",
    ]
    for clip, info in report.items():
        tags = json.dumps(list(info["metrics"].keys()))
        parts.append(f"initCase({json.dumps(clip)}, {tags});")
    parts.append("</script></div></body></html>")

    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    argv = [a for a in sys.argv[1:] if a != "--page-only"]
    page_only = "--page-only" in sys.argv
    out = Path(argv[0] if argv else EGS / "data_report/listening/v8_vs_v11")
    out.mkdir(parents=True, exist_ok=True)

    if page_only:
        # redraw the page from the measurements already on disk -- no model, no ffmpeg
        report = json.loads((out / "report.json").read_text())
        for clip, (_, headline, reading) in CASES.items():
            if clip in report:
                report[clip]["headline"] = headline
                report[clip]["reading"] = reading
        (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        build_page(report, out)
        print(f"page: {out / 'index.html'}")
        return

    windows = json.loads((CASES_DIR / "windows.json").read_text())
    models = {k: load_model(str(CONFIG), str(p), torch.device("cpu")) for k, p in CKPTS.items()}
    report: dict = {}

    for clip, (excerpt, headline, reading) in CASES.items():
        spec = windows[clip]
        raw, _ = AudioIO.open(f_path=str(CASES_DIR / f"{clip}_raw.wav"), target_lvl=None, resample_to=SR)
        raw = raw.view(1, -1)
        systems = {"mix": raw}
        for tag, model in models.items():
            systems[tag] = enhance(model, raw)
        ref_path = CASES_DIR / f"{clip}_qvf22.wav"
        if ref_path.is_file():
            ref, _ = AudioIO.open(f_path=str(ref_path), target_lvl=None, resample_to=SR)
            systems["qvf22"] = ref.view(1, -1)

        floor = spec["floor_dbfs"]
        near_ref = spec.get("near_ref_dbfs", float("nan"))
        keep_spans, supp_spans = spec.get("keep", []), spec.get("suppress", [])
        n_full = raw.shape[-1]
        keep_in = measure_spans(raw, keep_spans, n_full)
        supp_in = measure_spans(raw, supp_spans, n_full)
        headroom = supp_in - floor if supp_in == supp_in else float("nan")

        metrics = {}
        for tag, wav in systems.items():
            n = min(wav.shape[-1], n_full)
            keep_out = measure_spans(wav, keep_spans, n)
            supp_out = measure_spans(wav, supp_spans, n)
            row = {}
            if keep_spans:
                row["keep_out"] = r2(keep_out)
                row["keep"] = r2(keep_out - keep_in)
            if supp_spans:
                row["supp_out"] = r2(supp_out)
                row["reduc"] = r2(supp_out - supp_in)
                row["residual"] = r2(supp_out - floor)
                row["sir_out"] = r2(near_ref - supp_out)
            row["verdict"] = verdict(row.get("keep"), row.get("reduc"),
                                     row.get("residual"), r2(headroom))
            metrics[tag] = row

        per_span = []
        for kind, spans in (("keep", keep_spans), ("suppress", supp_spans)):
            for i, (t0, t1) in enumerate(spans, start=1):
                base = measure_spans(raw, [[t0, t1]], n_full)
                by_system = {}
                for tag, wav in systems.items():
                    if tag == "mix":
                        continue
                    out_db = measure_spans(wav, [[t0, t1]], min(wav.shape[-1], n_full))
                    by_system[tag] = r2(out_db - base)
                per_span.append({"kind": kind, "i": i, "t0": t0, "t1": t1, "by_system": by_system})

        a, b = (0.0, n_full / SR) if excerpt is None else excerpt
        i0, j0 = int(a * SR), int(b * SR)
        clipdir = out / clip
        clipdir.mkdir(exist_ok=True)
        panels, sizes = [], {}
        for tag, wav in systems.items():
            cut = wav[..., i0:j0]
            sizes[tag] = write_audio(cut, clipdir / tag)
            panels.append((tag if tag != "qvf22" else "QVF2.2 reference", cut.view(-1).numpy()))
        shifted = {kind: [[max(0.0, s - a), min(b - a, e - a)]
                          for s, e in spec.get(kind, []) if e > a and s < b]
                   for kind in ("keep", "suppress")}
        render_png(panels, shifted, (j0 - i0) / SR, clipdir / "spec.png")

        report[clip] = {
            "headline": headline,
            "reading": reading,
            "role": spec.get("role", ""),
            "group": spec.get("group"),
            "held_out": bool(spec.get("held_out", False)),
            "sentinel": bool(spec.get("sentinel", False)),
            "floor_dbfs": floor,
            "near_ref_dbfs": near_ref,
            "headroom_db": r2(headroom),
            "duration_s": round(n_full / SR, 3),
            "excerpt_s": [round(a, 3), round(b, 3)],
            "spans_full": {"keep": keep_spans, "suppress": supp_spans},
            "spans_excerpt": shifted,
            "metrics": metrics,
            "per_span": per_span,
            "mp3_bytes": sizes,
        }
        print(f"[done] {clip}", flush=True)

    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    build_page(report, out)
    print(f"\npage: {out / 'index.html'}")


if __name__ == "__main__":
    main()
