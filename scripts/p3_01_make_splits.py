"""Step 1 - build ONE source-video-disjoint train/val/test split over all kept
curated clips, then tag each clip with the tracks it belongs to.

Reads : thesis_dataset/logs/clips_manifest.csv  (existing curation, not modified)
Writes: thesis_p3/metadata/p3_clip_splits.csv
        thesis_p3/metadata/p3_split_summary.csv

Tracks
  A  realistic-noise      : every kept clip (all lighting)                -> cap TRACK_A_MAX_CLIPS
  B  day->night low-light : kept + lighting=day + quality=good
                            + condition in {clear, shadow}               -> cap TRACK_B_MAX_CLIPS
  C  real-world test      : kept + lighting in {evening, night}
                            + clip's source video is in the TEST split   -> cap TRACK_C_MAX_CLIPS

Because the split is assigned once per source video, no clip, scene or frame
can leak across train/val/test in any track.
"""

import csv
import random
from collections import Counter, defaultdict

from p3_config import (
    CLIPS_MANIFEST, CLIP_SPLITS_CSV, SPLIT_SUMMARY_CSV, METADATA_DIR,
    SPLIT_RATIOS, SPLIT_NAMES, SPLIT_SEED,
    TRACK_A_MAX_CLIPS, TRACK_B_MAX_CLIPS, TRACK_C_MAX_CLIPS,
)


def norm(v: str) -> str:
    return " ".join((v or "").strip().lower().split())


def load_kept_clips() -> list[dict]:
    with CLIPS_MANIFEST.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    kept = []
    for r in rows:
        if norm(r.get("keep_manual")) != "yes":
            continue
        # a handful of manifest rows have a bad clip_name; trust the path
        r["clip_name"] = r["clip_path"].replace("\\", "/").split("/")[-1]
        r["lighting_final"] = norm(r.get("lighting_final"))
        r["condition_final"] = norm(r.get("condition_final"))
        r["quality_final"] = norm(r.get("quality_final"))
        kept.append(r)
    return kept


# --------------------------------------------------------------------------- #
# source-disjoint stratified split (adapted from Phase 2 script 04)
# --------------------------------------------------------------------------- #
def label_counts(rows, col):
    return Counter(norm(r.get(col)) or "unlabeled" for r in rows)


def score_assignment(assign, stats, total, global_counts, strat_cols, required):
    per = {s: {"count": 0, "lighting": Counter(), "condition": Counter()} for s in SPLIT_NAMES}
    for src, s in assign.items():
        per[s]["count"] += stats[src]["count"]
        per[s]["lighting"].update(stats[src]["lighting"])
        per[s]["condition"].update(stats[src]["condition"])
    for s in SPLIT_NAMES:
        if per[s]["count"] == 0:
            return float("inf")
    for col, labels in required.items():
        for lab in labels:
            for s in SPLIT_NAMES:
                if per[s][col][lab] == 0:
                    return float("inf")
    score = 0.0
    for s, ratio in SPLIT_RATIOS.items():
        tgt = total * ratio
        score += 3.0 * ((per[s]["count"] - tgt) ** 2) / max(tgt, 1)
    for col in strat_cols:
        for lab, lab_total in global_counts[col].items():
            if not lab_total:
                continue
            for s, ratio in SPLIT_RATIOS.items():
                tgt = lab_total * ratio
                score += 0.5 * ((per[s][col][lab] - tgt) ** 2) / max(tgt, 1)
    return score


def make_split(rows, strat_cols, iterations=400_000):
    groups = defaultdict(list)
    for r in rows:
        groups[r["source_video"]].append(r)
    stats = {
        src: {"count": len(rs), "lighting": label_counts(rs, "lighting_final"),
              "condition": label_counts(rs, "condition_final")}
        for src, rs in groups.items()
    }
    sources = sorted(groups, key=lambda s: stats[s]["count"], reverse=True)
    total = len(rows)
    global_counts = {"lighting": label_counts(rows, "lighting_final"),
                     "condition": label_counts(rows, "condition_final")}

    # a label must appear in >=3 source videos before we force it into every split
    src_by_label = defaultdict(set)
    for src, rs in groups.items():
        for lab in label_counts(rs, "lighting_final"):
            src_by_label[lab].add(src)
    required = {"lighting": {lab for lab, ss in src_by_label.items() if len(ss) >= 3}}

    rng = random.Random(SPLIT_SEED)
    # greedy seed
    counts = {s: 0 for s in SPLIT_NAMES}
    greedy = {}
    for src in sources:
        s = min(SPLIT_NAMES, key=lambda n: counts[n] - SPLIT_RATIOS[n] * total)
        greedy[src] = s
        counts[s] += stats[src]["count"]
    candidates = [greedy]
    for _ in range(iterations):
        assign, counts = {}, {s: 0 for s in SPLIT_NAMES}
        shuffled = sources[:]
        rng.shuffle(shuffled)
        for src in shuffled:
            if rng.random() < 0.18:
                s = rng.choice(SPLIT_NAMES)
            else:
                s = min(SPLIT_NAMES, key=lambda n: counts[n] - SPLIT_RATIOS[n] * total)
            assign[src] = s
            counts[s] += stats[src]["count"]
        candidates.append(assign)

    best, best_score = None, float("inf")
    for a in candidates:
        sc = score_assignment(a, stats, total, global_counts, strat_cols, required)
        if sc < best_score:
            best, best_score = a, sc
    if best is None:
        raise RuntimeError("no valid split found")
    return best, best_score


# --------------------------------------------------------------------------- #
def diverse_cap(rows, cap, seed):
    """Deterministically keep <=cap rows, spreading across source/lighting/condition."""
    if not cap or len(rows) <= cap:
        return {r["clip_name"] for r in rows}
    rng = random.Random(seed)
    keyed = [(rng.random(), r) for r in rows]
    chosen, names = [], set()
    src_c, lit_c, cond_c = Counter(), Counter(), Counter()
    while len(chosen) < cap:
        cands = [it for it in keyed if it[1]["clip_name"] not in names]
        cands.sort(key=lambda it: (src_c[it[1]["source_video"]],
                                   lit_c[it[1]["lighting_final"]],
                                   cond_c[it[1]["condition_final"]], it[0]))
        _, r = cands[0]
        chosen.append(r); names.add(r["clip_name"])
        src_c[r["source_video"]] += 1
        lit_c[r["lighting_final"]] += 1
        cond_c[r["condition_final"]] += 1
    return names


def main() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    kept = load_kept_clips()
    print(f"kept clips: {len(kept)}")

    assign, score = make_split(kept, strat_cols=["lighting", "condition"])
    print(f"split score: {score:.3f}  (lower = better balanced)")
    src_split = {src: assign[src] for src in {r['source_video'] for r in kept}}

    for r in kept:
        r["split"] = assign[r["source_video"]]

    # track eligibility
    #   A  full benchmark pool (all splits)  -> training happens here + its own Gaussian val/test
    #   B  held-out low-light eval  (val+test day clips only, so no clip is trained on)
    #   C  held-out real-world eval (test evening/night clips only)
    a_pool = kept
    b_pool = [r for r in kept if r["lighting_final"] == "day"
              and r["quality_final"] == "good"
              and r["condition_final"] in {"clear", "shadow"}
              and r["split"] in {"val", "test"}]
    c_pool = [r for r in kept if r["lighting_final"] in {"evening", "night"}
              and r["split"] == "test"]

    a_names = diverse_cap(a_pool, TRACK_A_MAX_CLIPS, SPLIT_SEED + 1)
    b_names = diverse_cap(b_pool, TRACK_B_MAX_CLIPS, SPLIT_SEED + 2)
    c_names = diverse_cap(c_pool, TRACK_C_MAX_CLIPS, SPLIT_SEED + 3)

    out_cols = ["clip_name", "clip_path", "source_video", "split",
                "lighting_final", "condition_final", "quality_final",
                "clip_duration_sec", "in_track_a", "in_track_b", "in_track_c"]
    out_rows = []
    for r in kept:
        out_rows.append({
            "clip_name": r["clip_name"],
            "clip_path": r["clip_path"].replace("\\", "/"),
            "source_video": r["source_video"],
            "split": r["split"],
            "lighting_final": r["lighting_final"],
            "condition_final": r["condition_final"],
            "quality_final": r["quality_final"],
            "clip_duration_sec": r.get("clip_duration_sec", ""),
            "in_track_a": int(r["clip_name"] in a_names),
            "in_track_b": int(r["clip_name"] in b_names),
            "in_track_c": int(r["clip_name"] in c_names),
        })

    with CLIP_SPLITS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(out_rows)

    # summary
    summ_rows = []
    def block(name, rows):
        for s in SPLIT_NAMES:
            sr = [r for r in rows if r["split"] == s]
            lit = label_counts(sr, "lighting_final")
            con = label_counts(sr, "condition_final")
            summ_rows.append({
                "track": name, "split": s, "clips": len(sr),
                "source_videos": len({r["source_video"] for r in sr}),
                "day": lit["day"], "evening": lit["evening"], "night": lit["night"],
                "clear": con["clear"], "shadow": con["shadow"], "blur": con["blur"],
                "glare": con["glare"], "rain": con["rain"],
            })
    block("all_kept", kept)
    block("track_a", [r for r in kept if r["clip_name"] in a_names])
    block("track_b", [r for r in kept if r["clip_name"] in b_names])
    block("track_c", [r for r in kept if r["clip_name"] in c_names])

    with SPLIT_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ_rows[0].keys()))
        w.writeheader()
        w.writerows(summ_rows)

    print(f"\nsource videos per split: "
          f"{dict(Counter(src_split.values()))}")
    print(f"\nwrote {CLIP_SPLITS_CSV}")
    print(f"wrote {SPLIT_SUMMARY_CSV}\n")
    hdr = f"{'track':9s} {'split':5s} {'clips':>5s} {'src':>3s}  day/eve/night   clear/shad/blur/glare/rain"
    print(hdr); print("-" * len(hdr))
    for r in summ_rows:
        print(f"{r['track']:9s} {r['split']:5s} {r['clips']:5d} {r['source_videos']:3d}  "
              f"{r['day']:3d}/{r['evening']:3d}/{r['night']:3d}      "
              f"{r['clear']:3d}/{r['shadow']:3d}/{r['blur']:3d}/{r['glare']:3d}/{r['rain']:3d}")


if __name__ == "__main__":
    main()
