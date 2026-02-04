import pandas as pd
import re
import difflib
import json
import os
from datetime import date

# ============================================================
# Conversation state (for yes/no clarifications)
# ============================================================
STATE = {"pending": None}

YES_WORDS = {"y", "yes", "yeah", "yep", "true", "correct", "sure", "ok", "okay"}
NO_WORDS  = {"n", "no", "nope", "nah", "false"}

# Optional aliases (add more anytime)
ALIASES = {
    "o malley": ["sean omalley", "sean o'malley"],
    "omalley": ["sean omalley", "sean o'malley"],
    "sean o malley": ["sean omalley", "sean o'malley"],
    "song yadong": ["yadong song"],
    "yadong": ["song yadong"],
    "umar": ["umar nurmagomedov"],
}

# ============================================================
# Weight class ordering (exactly as you stated)
# ============================================================
MALE_WC_ORDER = [
    "flyweight",
    "bantamweight",
    "featherweight",
    "lightweight",
    "welterweight",
    "middleweight",
    "light heavyweight",
    "heavyweight",
]

FEMALE_WC_ORDER = [
    "women's strawweight",
    "women's flyweight",
    "women's bantamweight",
    "women's featherweight",
]

WC_RANK = {w: i for i, w in enumerate(MALE_WC_ORDER)}
WC_RANK.update({w: i for i, w in enumerate(FEMALE_WC_ORDER)})

def _norm_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"['’]", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _norm_wc(wc: str) -> str:
    wc = _norm_text(wc)
    wc = wc.replace("womens", "women's")
    wc = wc.replace("women s", "women's")
    wc = wc.replace("lightheavyweight", "light heavyweight")
    wc = wc.replace("light-heavyweight", "light heavyweight")
    return wc.strip()

def _wc_rank(wc: str):
    wc = _norm_wc(wc)
    return WC_RANK.get(wc, None)

def _wc_label(fid: int, fighters_df):
    row = fighters_df[fighters_df["Fighter_ID"] == fid]
    if len(row) == 0:
        return None
    if "Weight_Class" not in row.columns:
        return None
    return str(row.iloc[0]["Weight_Class"])

# ============================================================
# Persistent caches (prevents flip-flopping across restarts)
# ============================================================
CACHE_FILE = "cache.json"
PRED_CACHE = {}
PROP_CACHE = {}
PRED_FORMAT_VERSION = "2"


def _load_cache():
    global PRED_CACHE, PROP_CACHE
    if not os.path.exists(CACHE_FILE):
        return
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        PRED_CACHE = data.get("pred", {}) or {}
        PROP_CACHE = data.get("prop", {}) or {}
    except Exception:
        PRED_CACHE = {}
        PROP_CACHE = {}

def _save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"pred": PRED_CACHE, "prop": PROP_CACHE}, f, indent=2)
    except Exception:
        pass

def _cache_key(*parts):
    return "|".join(str(p) for p in parts)

_load_cache()

# ============================================================
# Numeric helpers
# ============================================================
def _to_num(x):
    return pd.to_numeric(x, errors="coerce")

def _parse_date(x):
    return pd.to_datetime(x, errors="coerce")

def _parse_control_seconds(x):
    if pd.isna(x):
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return float("nan")
    n = pd.to_numeric(s, errors="coerce")
    if pd.notna(n):
        return float(n)
    m = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", s)
    if m:
        return float(int(m.group(1)) * 60 + int(m.group(2)))
    return float("nan")

def _safe_div(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return float("nan")
    return float(a) / float(b)

def _format_num(x, decimals=3):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{decimals}f}"

def _format_seconds(sec):
    if pd.isna(sec):
        return "N/A"
    sec = float(sec)
    m = int(sec // 60)
    s = int(round(sec - m * 60))
    return f"{m}:{s:02d}"

def _pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x*100:.1f}%"

def _norm_edge(a, b, eps=1e-9):
    if pd.isna(a) or pd.isna(b):
        return float("nan")
    denom = abs(a) + abs(b) + eps
    return (a - b) / denom

def _tier_from_strength(x_abs: float):
    if x_abs >= 1.00:
        return "Strong"
    if x_abs >= 0.60:
        return "Moderate"
    if x_abs >= 0.30:
        return "Lean"
    return "Toss-up"

def _arrow(delta, small=0.05, med=0.15):
    if pd.isna(delta):
        return "—"
    if delta >= med:
        return "↑↑"
    if delta >= small:
        return "↑"
    if delta <= -med:
        return "↓↓"
    if delta <= -small:
        return "↓"
    return "→"

# ============================================================
# Load CSVs
# ============================================================
print("\nLoading CSV files...")
fighters = pd.read_csv("Fighters.csv")
fights = pd.read_csv("Fights.csv")
stats = pd.read_csv("Fight_Stats.csv")
print("✅ Loaded successfully!\n")

fighters["Fighter_Name"] = fighters["Fighter_Name"].astype(str).str.strip()
fighters["name_clean"] = fighters["Fighter_Name"].astype(str).apply(_norm_text)
# ============================================================
# Gist matcher: detect 2 fighter names anywhere in a long message
# (Conservative: only triggers when exactly 2 fighters are found)
# ============================================================
def _extract_two_fighters_anywhere(text: str):
    """
    Returns (fid1, fid2) if exactly two distinct fighter IDs are confidently
    present in the text (by substring match on name_clean / aliases). Otherwise None.
    """
    lowc = _norm_text(text)

    found = []
    seen = set()

    # 1) Exact substring match against canonical cleaned full names
    # (This is fast and very safe.)
    for _, row in fighters[["Fighter_ID", "name_clean"]].iterrows():
        name_clean = row["name_clean"]
        if not name_clean:
            continue
        if name_clean in lowc:
            fid = int(row["Fighter_ID"])
            if fid not in seen:
                seen.add(fid)
                found.append(fid)
                if len(found) > 2:
                    return None  # too many fighters mentioned; don't guess

    # 2) Also allow alias keys to match (e.g. "o malley") but map via find_fighter_id
    # Only if we still haven't found 2.
    if len(found) < 2:
        for alias_key, alias_vals in ALIASES.items():
            ak = _norm_text(alias_key)
            if ak and ak in lowc:
                fid = find_fighter_id(alias_key, allow_ask=False)
                if isinstance(fid, int) and fid not in seen:
                    seen.add(fid)
                    found.append(fid)
                    if len(found) > 2:
                        return None

    return (found[0], found[1]) if len(found) == 2 else None

# ============================================================
# Fighter helpers
# ============================================================
def fighter_name(fid: int):
    row = fighters[fighters["Fighter_ID"] == fid]
    return str(row.iloc[0]["Fighter_Name"]) if len(row) else str(fid)

def fighter_row(fid: int):
    row = fighters[fighters["Fighter_ID"] == fid]
    if len(row) == 0:
        return None
    return row.iloc[0]

# ============================================================
# Smart fighter lookup (fuzzy + ambiguity support)
# ============================================================
def find_fighter_id(name: str, allow_ask=False):
    query_raw = str(name).strip()
    q = _norm_text(query_raw)

    if q in ALIASES:
        for alt in ALIASES[q]:
            fid = find_fighter_id(alt, allow_ask=False)
            if isinstance(fid, int):
                return fid

    exact = fighters[fighters["name_clean"] == q]
    if len(exact) == 1:
        return int(exact.iloc[0]["Fighter_ID"])

    if q:
        sub = fighters[fighters["name_clean"].str.contains(re.escape(q), na=False)]
        if len(sub) == 1:
            return int(sub.iloc[0]["Fighter_ID"])

    tokens = set(q.split())
    scored = []
    for _, row in fighters.iterrows():
        name_clean = row["name_clean"]
        parts = name_clean.split()
        if not parts:
            continue

        score = 0.0
        if len(parts) >= 2:
            last = parts[-1]
            first = parts[0]
            if last in tokens:
                score += 2.0
            if first in tokens:
                score += 1.0
            if first in tokens and last in tokens:
                score += 2.0
        else:
            if parts[0] in tokens:
                score += 1.0

        fuzzy = difflib.SequenceMatcher(None, q, name_clean).ratio()
        score += 3.0 * fuzzy

        scored.append((score, int(row["Fighter_ID"]), str(row["Fighter_Name"])))

    if not scored:
        return None

    scored.sort(reverse=True, key=lambda x: x[0])
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None

    best_score = best[0]
    second_score = second[0] if second else -999

    if best_score >= 3.8 and (best_score - second_score) >= 0.35:
        return best[1]

    options = [x for x in scored[:8] if x[0] >= (best_score - 0.45)]
    if allow_ask and len(options) >= 2:
        return "__AMBIG__", options

    if best_score >= 3.2:
        return best[1]

    return None

def _ask_clarify(options, build_yes_command, title=""):
    top = options[0]
    top_name = top[2]
    yes_cmd = build_yes_command(top[1])

    alt_names = [o[2] for o in options[1:6]]
    alt_text = ", ".join(alt_names) if alt_names else "(no other close matches)"

    question = f"{title}Did you mean **{top_name}**? (yes/no)"
    no_prompt = f"If no, type the correct full name. Closest matches: {alt_text}"

    STATE["pending"] = {
        "question": question,
        "yes_command": yes_cmd,
        "no_prompt": no_prompt,
    }
    return question

# ============================================================
# Fight filtering (completed vs upcoming)
# ============================================================
fights2 = fights.copy()
fights2["Fight_Date"] = _parse_date(fights2.get("Fight_Date"))
fights2["Winner_Fighter_ID"] = _to_num(fights2.get("Winner_Fighter_ID"))
today_dt = pd.Timestamp(date.today())
# Ensure ID columns are numeric so upcoming fight matching works
for col in ["Fight_ID", "Fighter_A_ID", "Fighter_B_ID", "Rounds_Scheduled"]:
    if col in fights2.columns:
        fights2[col] = _to_num(fights2.get(col))

completed_fights = fights2[fights2["Winner_Fighter_ID"].notna()].copy()
upcoming_fights = fights2[(fights2["Fight_Date"].notna()) & (fights2["Fight_Date"] >= today_dt)].copy()

# ============================================================
# Clean stats + join to completed fights
# ============================================================
stats2 = stats.copy()
stats2["Fight_ID"] = _to_num(stats2.get("Fight_ID"))
stats2["Fighter_ID"] = _to_num(stats2.get("Fighter_ID"))
stats2["Opponent_ID"] = _to_num(stats2.get("Opponent_ID"))

for col in [
    "Significant_Strikes_Landed",
    "Significant_Strikes_Attempted",
    "Takedowns_Landed",
    "Takedowns_Attempted",
    "Knockdowns",
    "Duration",
    "Fight_Duration_Seconds",
]:
    if col in stats2.columns:
        stats2[col] = _to_num(stats2[col])

if "Control_Time" in stats2.columns:
    stats2["Control_Time_Seconds"] = stats2["Control_Time"].apply(_parse_control_seconds)
else:
    stats2["Control_Time_Seconds"] = float("nan")

join_cols = ["Fight_ID", "Fight_Date", "Fight_Duration_Seconds", "Method", "Round_Ended", "Rounds_Scheduled", "Weight_Class"]
join_cols = [c for c in join_cols if c in completed_fights.columns]

stats_completed = stats2.merge(
    completed_fights[join_cols],
    on="Fight_ID",
    how="left",
)

stats_completed = stats_completed[stats_completed["Fight_Date"].notna()].copy()
stats_completed["Fight_Duration_Seconds"] = _to_num(stats_completed.get("Fight_Duration_Seconds"))
if "Rounds_Scheduled" in stats_completed.columns:
    stats_completed["Rounds_Scheduled"] = _to_num(stats_completed.get("Rounds_Scheduled"))

def _get_completed_rows_for_fighter(fid: int):
    rows = stats_completed[stats_completed["Fighter_ID"] == fid].copy()
    rows = rows.sort_values("Fight_Date", ascending=False)
    return rows

def _last_n(rows: pd.DataFrame, n):
    if n is None:
        return rows
    return rows.head(int(n))

def _per_fight_minutes(rows: pd.DataFrame):
    m = rows["Fight_Duration_Seconds"] / 60.0
    return m.replace([0, -0], float("nan"))

# ============================================================
# Defense join: opponent stats from same fight
# ============================================================
opp_cols = {
    "Significant_Strikes_Landed": "Opp_Sig_Landed",
    "Significant_Strikes_Attempted": "Opp_Sig_Att",
    "Takedowns_Landed": "Opp_TD_Landed",
    "Takedowns_Attempted": "Opp_TD_Att",
    "Knockdowns": "Opp_KD",
    "Control_Time_Seconds": "Opp_Ctrl_Sec",
}

need_cols = ["Fight_ID", "Fighter_ID"] + [c for c in opp_cols.keys() if c in stats_completed.columns]
opp = stats_completed[need_cols].copy()
opp = opp.rename(columns={"Fighter_ID": "Opponent_ID", **{k: v for k, v in opp_cols.items() if k in opp.columns}})
stats_completed = stats_completed.merge(opp, on=["Fight_ID", "Opponent_ID"], how="left")

# ============================================================
# Metrics
# ============================================================
def _metrics_from_rows(rows: pd.DataFrame):
    out = {}
    if len(rows) == 0:
        return out

    mins_each = _per_fight_minutes(rows)
    mins_total = mins_each.sum(skipna=True)
    if pd.isna(mins_total) or mins_total == 0:
        mins_total = float("nan")

    sig_l = rows.get("Significant_Strikes_Landed", pd.Series(dtype=float)).sum(skipna=True)
    sig_a = rows.get("Significant_Strikes_Attempted", pd.Series(dtype=float)).sum(skipna=True)
    td_l  = rows.get("Takedowns_Landed", pd.Series(dtype=float)).sum(skipna=True)
    td_a  = rows.get("Takedowns_Attempted", pd.Series(dtype=float)).sum(skipna=True)
    kd    = rows.get("Knockdowns", pd.Series(dtype=float)).sum(skipna=True)
    ctrl  = rows.get("Control_Time_Seconds", pd.Series(dtype=float)).sum(skipna=True)

    sig_abs = rows.get("Opp_Sig_Landed", pd.Series(dtype=float)).sum(skipna=True)
    td_allowed = rows.get("Opp_TD_Landed", pd.Series(dtype=float)).sum(skipna=True)
    kd_taken = rows.get("Opp_KD", pd.Series(dtype=float)).sum(skipna=True)
    ctrl_allowed = rows.get("Opp_Ctrl_Sec", pd.Series(dtype=float)).sum(skipna=True)

    out["fights_n"] = len(rows)

    out["sig_pm"] = _safe_div(sig_l, mins_total)
    out["td_pm"]  = _safe_div(td_l, mins_total)
    out["td_p15"] = _safe_div(td_l, mins_total) * 15.0
    out["ctrl_pm_sec"] = _safe_div(ctrl, mins_total) * 60.0   # control seconds per minute
    out["ctrl_p15_sec"] = _safe_div(ctrl, mins_total) * 15.0
    out["kd_pm"]  = _safe_div(kd, mins_total)

    out["sig_abs_pm"] = _safe_div(sig_abs, mins_total)
    out["td_allowed_p15"] = _safe_div(td_allowed, mins_total) * 15.0
    out["ctrl_allowed_p15_sec"] = _safe_div(ctrl_allowed, mins_total) * 15.0

    out["sig_acc"] = _safe_div(sig_l, sig_a)
    out["td_acc"]  = _safe_div(td_l, td_a)

    out["avg_sig_per_fight"] = _safe_div(sig_l, len(rows))
    out["avg_td_per_fight"]  = _safe_div(td_l, len(rows))
    out["avg_ctrl_sec_per_fight"] = _safe_div(ctrl, len(rows))
    out["avg_time_sec"] = rows.get("Fight_Duration_Seconds", pd.Series(dtype=float)).mean(skipna=True)

    # Recent form pieces
    if "Result" in rows.columns:
        res = rows["Result"].astype(str).str.strip().str.upper()
        out["wins"] = int((res == "W").sum())
        out["losses"] = int((res == "L").sum())
        out["draws"] = int(((res == "D") | (res == "DRAW")).sum())
        out["ncs"] = int(((res == "NC") | (res == "NO CONTEST") | (res == "NO_CONTEST")).sum())

    # variability
    time_min_each = rows.get("Fight_Duration_Seconds", pd.Series(dtype=float)) / 60.0
    sig_pm_each = rows.get("Significant_Strikes_Landed", pd.Series(dtype=float)) / mins_each
    td_p15_each = (rows.get("Takedowns_Landed", pd.Series(dtype=float)) / mins_each) * 15.0
    out["std_time_min"] = time_min_each.std(skipna=True)
    out["std_sig_pm"] = sig_pm_each.std(skipna=True)
    out["std_td_p15"] = td_p15_each.std(skipna=True)

    return out

# ============================================================
# Blending: career + recent weight
# ============================================================
RECENT_N_DEFAULT = 5
RECENT_WEIGHT_DEFAULT = 0.60

def _blend(career, recent, w):
    if pd.isna(career) and pd.isna(recent):
        return float("nan")
    if pd.isna(career):
        return recent
    if pd.isna(recent):
        return career
    return (1-w)*career + w*recent

def _get_blended_metrics(fid: int, recent_n=RECENT_N_DEFAULT, recent_weight=RECENT_WEIGHT_DEFAULT):
    rows_all = _get_completed_rows_for_fighter(fid)
    rows_recent = _last_n(rows_all, recent_n)

    career = _metrics_from_rows(rows_all)
    recent = _metrics_from_rows(rows_recent)

    keys = list(set(career.keys()) | set(recent.keys()))
    blended = {}
    for k in keys:
        blended[k] = _blend(career.get(k, float("nan")), recent.get(k, float("nan")), recent_weight)

    return blended, career, recent

def _get_lastn_metrics(fid: int, last_n: int):
    rows = _last_n(_get_completed_rows_for_fighter(fid), last_n)
    return _metrics_from_rows(rows)

# ============================================================
# Upcoming fight helpers
# ============================================================
def _find_next_upcoming_for_fighter(fid: int):
    f = upcoming_fights.copy()
    cond = (f.get("Fighter_A_ID") == fid) | (f.get("Fighter_B_ID") == fid)
    cand = f[cond].copy()
    if len(cand) == 0:
        return None
    cand = cand.sort_values("Fight_Date", ascending=True)
    return cand.iloc[0]

def _upcoming_opponent(fid: int):
    # Make sure fid is always an int for comparisons
    try:
        fid = int(fid)
    except Exception:
        return None, None

    row = _find_next_upcoming_for_fighter(fid)
    if row is None:
        return None, None

    a = pd.to_numeric(row.get("Fighter_A_ID"), errors="coerce")
    b = pd.to_numeric(row.get("Fighter_B_ID"), errors="coerce")
    if pd.isna(a) or pd.isna(b):
        return None, None

    a = int(a)
    b = int(b)

    # IMPORTANT: only return an opponent if fid is actually in the matchup row
    if a == fid:
        return b, row
    if b == fid:
        return a, row

    # If the row doesn't actually contain fid, treat as "no upcoming fight"
    return None, None


def _resolve_rounds_for_upcoming(fid: int, opp_id: int):
    row = _find_next_upcoming_for_fighter(fid)
    rounds = 3
    if row is not None and "Rounds_Scheduled" in row.index:
        rs = pd.to_numeric(row.get("Rounds_Scheduled"), errors="coerce")
        if pd.notna(rs) and int(rs) in (3, 5):
            rounds = int(rs)
    return rounds

def _expected_fight_minutes(A: dict, B: dict, rounds_scheduled: int):
    sched_min = 5.0 * int(rounds_scheduled)
    a_t = A.get("avg_time_sec", float("nan"))
    b_t = B.get("avg_time_sec", float("nan"))
    if pd.isna(a_t): a_t = sched_min * 60
    if pd.isna(b_t): b_t = sched_min * 60
    exp_sec = min(sched_min * 60, 0.5 * a_t + 0.5 * b_t)
    return exp_sec / 60.0
def _snap_expected_minutes(: float, rounds_scheduled: int, snap_threshold: float = 0.85):
    sched_minutes = 5.0 * int(rounds_scheduled)
    if pd.isna():
        return float("nan")
    if  >= snap_threshold * sched_minutes:
        return sched_minutes
    return min(, sched_minutes)

# ============================================================
# Style tags + recent form/trajectory
# ============================================================
def style_tags(fid: int, last_n=5):
    rows_all = _get_completed_rows_for_fighter(fid)
    rows_recent = _last_n(rows_all, last_n)

    if len(rows_recent) == 0 and len(rows_all) == 0:
        return f"No completed fights found for {fighter_name(fid)}."

    recent = _metrics_from_rows(rows_recent) if len(rows_recent) else {}
    career = _metrics_from_rows(rows_all) if len(rows_all) else {}

    sig_pm = recent.get("sig_pm", career.get("sig_pm", float("nan")))
    td_p15 = recent.get("td_p15", career.get("td_p15", float("nan")))
    ctrl_p15 = recent.get("ctrl_p15_sec", career.get("ctrl_p15_sec", float("nan")))
    kd_pm = recent.get("kd_pm", career.get("kd_pm", float("nan")))
    sig_abs_pm = recent.get("sig_abs_pm", career.get("sig_abs_pm", float("nan")))
    td_allow = recent.get("td_allowed_p15", career.get("td_allowed_p15", float("nan")))

    tags = []

    # Striking style
    if pd.notna(sig_pm):
        if sig_pm >= 3.8:
            tags.append("High-volume striker")
        elif sig_pm >= 2.6:
            tags.append("Active striker")
        else:
            tags.append("Low-output striker")

    if pd.notna(kd_pm) and kd_pm >= 0.06:
        tags.append("Power threat (KDs)")

    # Grappling style
    if pd.notna(td_p15):
        if td_p15 >= 3.0:
            tags.append("Chain wrestler")
        elif td_p15 >= 1.5:
            tags.append("Wrestling threat")

    if pd.notna(ctrl_p15):
        if ctrl_p15 >= 240:  # 4:00 per 15
            tags.append("Heavy top control")
        elif ctrl_p15 >= 120:
            tags.append("Control-oriented")

    # Defense style
    if pd.notna(sig_abs_pm):
        if sig_abs_pm <= 2.0:
            tags.append("Tight striking defense")
        elif sig_abs_pm >= 3.6:
            tags.append("Hit a lot (high absorption)")

    if pd.notna(td_allow):
        if td_allow <= 1.0:
            tags.append("Strong TDD (low allowed)")
        elif td_allow >= 3.0:
            tags.append("Vulnerable to takedowns")

    # Make readable
    if not tags:
        tags = ["Insufficient data for style tags"]

    return f"STYLE TAGS — {fighter_name(fid)} (last {last_n} fights weighted)\n• " + "\n• ".join(tags)

def recent_form(fid: int, last_n=5):
    rows = _last_n(_get_completed_rows_for_fighter(fid), last_n)
    if len(rows) == 0:
        return f"No completed fights found for {fighter_name(fid)}."

    if "Result" in rows.columns:
        res = rows["Result"].astype(str).str.strip().str.upper().tolist()
        w = sum(r == "W" for r in res)
        l = sum(r == "L" for r in res)
        d = sum((r == "D") or (r == "DRAW") for r in res)
        nc = sum((r == "NC") or (r == "NO CONTEST") or (r == "NO_CONTEST") for r in res)
    else:
        res = []
        w = l = d = nc = 0

    # Trajectory: compare last_n stats to career stats
    all_rows = _get_completed_rows_for_fighter(fid)
    career = _metrics_from_rows(all_rows)
    recent = _metrics_from_rows(rows)

    def delta(key):
        return recent.get(key, float("nan")) - career.get(key, float("nan"))

    traj_lines = [
        f"Significant strikes/min: {_arrow(delta('sig_pm'))}",
        f"Takedowns/15: {_arrow(delta('td_p15'))}",
        f"Control/15: {_arrow(delta('ctrl_p15_sec'))}",
        f"Absorbed sig/min (lower better): {_arrow(-delta('sig_abs_pm'))}",
        f"TD allowed/15 (lower better): {_arrow(-delta('td_allowed_p15'))}",
    ]

    return "\n".join([
        f"RECENT FORM — {fighter_name(fid)} (last {last_n})",
        f"Results: {w}-{l}-{d} (NC: {nc})" + (f" | Sequence: {' '.join(res)}" if res else ""),
        "Trajectory vs career:",
        "• " + "\n• ".join(traj_lines)
    ])

# ============================================================
# Data health checks
# ============================================================
def data_health():
    issues = []

    # Required files/columns checks
    required_fighters = ["Fighter_ID", "Fighter_Name"]
    required_fights = ["Fight_ID", "Fight_Date", "Fighter_A_ID", "Fighter_B_ID"]
    required_stats = ["Fight_ID", "Fighter_ID", "Opponent_ID"]

    for c in required_fighters:
        if c not in fighters.columns:
            issues.append(f"Fighters.csv missing column: {c}")
    for c in required_fights:
        if c not in fights.columns:
            issues.append(f"Fights.csv missing column: {c}")
    for c in required_stats:
        if c not in stats.columns:
            issues.append(f"Fight_Stats.csv missing column: {c}")

    # Duplicate IDs
    if "Fighter_ID" in fighters.columns:
        dup = fighters["Fighter_ID"].duplicated().sum()
        if dup:
            issues.append(f"Duplicate Fighter_ID rows: {dup}")

    if "Fight_ID" in fights.columns:
        dup = fights["Fight_ID"].duplicated().sum()
        if dup:
            issues.append(f"Duplicate Fight_ID rows: {dup}")

    # Stats rows with missing opponent or fighter
    if "Fighter_ID" in stats.columns and "Opponent_ID" in stats.columns:
        missing_opp = stats["Opponent_ID"].isna().sum()
        if missing_opp:
            issues.append(f"Fight_Stats rows missing Opponent_ID: {missing_opp}")

    # Completed fights that have no stats rows
    if "Fight_ID" in completed_fights.columns:
        cf_ids = set(completed_fights["Fight_ID"].tolist())
        st_ids = set(stats["Fight_ID"].dropna().tolist())
        missing_stats = len(cf_ids - st_ids)
        if missing_stats:
            issues.append(f"Completed fights with NO stats rows in Fight_Stats.csv: {missing_stats}")

    if not issues:
        return "DATA HEALTH — Looks good ✅ (no obvious issues found)."

    return "DATA HEALTH — Issues found:\n• " + "\n• ".join(issues)

# ============================================================
# Stat functions (kept + expanded)
# ============================================================
def props_over_under_by_id(fid: int, line: float, stat: str, opp_name: str = None):
    stat = (stat or "").strip().lower()

    # Resolve opponent (explicit or upcoming)
    if opp_name:
        opp_res = find_fighter_id(opp_name, allow_ask=True)
        if isinstance(opp_res, tuple) and opp_res[0] == "__AMBIG__":
            return "__AMBIG_OPP__", opp_res[1]
        if not isinstance(opp_res, int):
            return f"I couldn't match the opponent name: {opp_name}"
        opp_id = opp_res
        opp_row = None
    else:
        opp_id, opp_row = _upcoming_opponent(fid)
        if opp_id is None:
            STATE["pending"] = {
                "kind": "props_need_opp",
                "fid": fid,
                "stat": stat,
                "line": float(line),
            }
            return f"I don't see an upcoming fight for {fighter_name(fid)}. Who is the opponent?"

    
    # Rounds
    rounds = 3
    if opp_row is not None:
        rs = pd.to_numeric(opp_row.get("Rounds_Scheduled"), errors="coerce")
        if pd.notna(rs) and int(rs) in (3, 5):
            rounds = int(rs)

    # === SINGLE SOURCE OF TRUTH FOR EXPECTED TIME ===
    A, _, _ = _get_blended_metrics(fid)
    B, _, _ = _get_blended_metrics(opp_id)

    # Use the SAME fight-ending time that the prediction uses (finish time if finish, 15/25 if decision)
    pred_text = predict(fid, opp_id, last_n_override=None)

    m = re.search(r"Expected fight time:\s*([0-9]+(\.[0-9]+)?)", pred_text)
    if m:
        exp_minutes = float(m.group(1))
    else:
        # Fallback (should almost never happen)
        exp_minutes = _snap_expected_minutes(_expected_fight_minutes(A, B, rounds), rounds)


    # Expected rates
    exp_sigpm = _expected_rate(A.get("sig_pm"), B.get("sig_abs_pm"))
    exp_tdp15 = _expected_rate(A.get("td_p15"), B.get("td_allowed_p15"))

    exp_sig_total = exp_sigpm * exp_minutes if pd.notna(exp_sigpm) else float("nan")
    exp_td_total = exp_tdp15 * (exp_minutes / 15.0) if pd.notna(exp_tdp15) else float("nan")

    if stat in {"sig", "significant strikes"}:
        exp = exp_sig_total
        label = "significant strikes"
    elif stat in {"td", "takedowns"}:
        exp = exp_td_total
        label = "takedowns"
    elif stat in {"minutes", "time"}:
        exp = exp_minutes
        label = "minutes"
    else:
        return "Stat must be sig, td, or minutes."

    diff = exp - float(line)
    pick = "Over" if diff > 0 else "Under" if diff < 0 else "Push"

    return (
        f"PROP — {fighter_name(fid)} vs {fighter_name(opp_id)}\n"
        f"{pick} {line} {label}\n"
        f"Model expected: {exp:.2f} {label} (expected fight time {exp_minutes:.2f} min)"
    )


def last_fight_stat_by_id(fid: int, metric: str = "summary"):
    rows = _get_completed_rows_for_fighter(fid)  # already sorted most-recent first
    if len(rows) == 0:
        return f"No completed fights found for {fighter_name(fid)}."
    r = rows.iloc[0]

    # Opponent + date
    opp_id = r.get("Opponent_ID")
    opp_name = fighter_name(int(opp_id)) if pd.notna(opp_id) else "Unknown"
    fdate = r.get("Fight_Date")
    date_str = str(pd.to_datetime(fdate).date()) if pd.notna(fdate) else "Unknown date"

    # Pull raw values safely
    sig_l = r.get("Significant_Strikes_Landed", float("nan"))
    sig_a = r.get("Significant_Strikes_Attempted", float("nan"))
    td_l = r.get("Takedowns_Landed", float("nan"))
    td_a = r.get("Takedowns_Attempted", float("nan"))
    kd = r.get("Knockdowns", float("nan"))
    ctrl = r.get("Control_Time_Seconds", float("nan"))
    dur = r.get("Fight_Duration_Seconds", float("nan"))
    res = str(r.get("Result", "")).strip()

    # Helpers for formatting
    def nint(x):
        return "N/A" if pd.isna(x) else str(int(x))

    if metric in {"sig", "sig strikes", "significant strikes"}:
        return f"LAST FIGHT — {fighter_name(fid)} vs {opp_name} ({date_str})\nSignificant strikes: {nint(sig_l)}/{nint(sig_a)}"
    if metric in {"td", "takedown", "takedowns"}:
        return f"LAST FIGHT — {fighter_name(fid)} vs {opp_name} ({date_str})\nTakedowns: {nint(td_l)}/{nint(td_a)}"
    if metric in {"control", "control time"}:
        return f"LAST FIGHT — {fighter_name(fid)} vs {opp_name} ({date_str})\nControl time: {_format_seconds(ctrl)}"
    if metric in {"minutes", "time", "duration"}:
        return f"LAST FIGHT — {fighter_name(fid)} vs {opp_name} ({date_str})\nFight time: {_format_seconds(dur)}"
    if metric in {"knockdowns", "kd"}:
        return f"LAST FIGHT — {fighter_name(fid)} vs {opp_name} ({date_str})\nKnockdowns: {nint(kd)}"

    # Default summary
    return f"LAST FIGHT — {fighter_name(fid)} vs {opp_name} ({date_str})\nResult: {res if res else 'N/A'} | Sig: {nint(sig_l)}/{nint(sig_a)} | TD: {nint(td_l)}/{nint(td_a)} | Ctrl: {_format_seconds(ctrl)} | Time: {_format_seconds(dur)} | KD: {nint(kd)}"



def td_def_by_id(fid: int, last_n=None):
    rows = _rows_for(fid, last_n)
    if len(rows) == 0:
        return f"No completed fights found for {fighter_name(fid)}."

    td_att_against = rows.get("Opp_TD_Att", pd.Series(dtype=float)).sum(skipna=True)
    td_landed_against = rows.get("Opp_TD_Landed", pd.Series(dtype=float)).sum(skipna=True)

    if pd.isna(td_att_against) or td_att_against == 0:
        return f"{fighter_name(fid)} has no opponent takedown attempts recorded."

    tdd = 1.0 - (td_landed_against / td_att_against)
    tdd = max(0.0, min(1.0, tdd))

    return f"TAKEDOWN DEFENSE — {fighter_name(fid)}{' (last ' + str(last_n) + ')' if last_n else ''}\nTDD: {_pct(tdd)} (opponents landed {int(td_landed_against)}/{int(td_att_against)} takedowns)"

def last_n_fights_stat_by_id(fid: int, n: int, metric: str):
    rows = _get_completed_rows_for_fighter(fid)
    if len(rows) == 0:
        return f"No completed fights found for {fighter_name(fid)}."

    rows = rows.head(n)
    if len(rows) == 0:
        return f"No completed fights found for {fighter_name(fid)}."

    sig_l = rows.get("Significant_Strikes_Landed", pd.Series(dtype=float)).sum(skipna=True)
    sig_a = rows.get("Significant_Strikes_Attempted", pd.Series(dtype=float)).sum(skipna=True)
    td_l = rows.get("Takedowns_Landed", pd.Series(dtype=float)).sum(skipna=True)
    td_a = rows.get("Takedowns_Attempted", pd.Series(dtype=float)).sum(skipna=True)
    ctrl = rows.get("Control_Time_Seconds", pd.Series(dtype=float)).sum(skipna=True)
    dur = rows.get("Fight_Duration_Seconds", pd.Series(dtype=float)).sum(skipna=True)
    kd = rows.get("Knockdowns", pd.Series(dtype=float)).sum(skipna=True)

    fights_n = len(rows)

    def avg(x):
        return "N/A" if pd.isna(x) else f"{x / fights_n:.2f}"

    if metric == "sig":
        return f"LAST {fights_n} FIGHTS — {fighter_name(fid)}\nAvg significant strikes: {avg(sig_l)}"
    if metric == "td":
        return f"LAST {fights_n} FIGHTS — {fighter_name(fid)}\nAvg takedowns: {avg(td_l)}"
    if metric == "control":
        return f"LAST {fights_n} FIGHTS — {fighter_name(fid)}\nAvg control time: {_format_seconds(ctrl / fights_n)}"
    if metric == "minutes":
        return f"LAST {fights_n} FIGHTS — {fighter_name(fid)}\nAvg fight time: {_format_seconds(dur / fights_n)}"
    if metric == "knockdowns":
        return f"LAST {fights_n} FIGHTS — {fighter_name(fid)}\nAvg knockdowns: {avg(kd)}"

    return (
        f"LAST {fights_n} FIGHTS — {fighter_name(fid)}\n"
        f"Avg sig: {avg(sig_l)} | Avg TD: {avg(td_l)} | "
        f"Avg ctrl: {_format_seconds(ctrl / fights_n)} | "
        f"Avg time: {_format_seconds(dur / fights_n)} | "
        f"Avg KD: {avg(kd)}"
    )


# ============================================================
# Record helpers (W-L-D (NC)) + total UFC fights
# ============================================================
def _norm_result_token(x: str):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().upper().replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()

    if s in {"W", "WIN", "WINS"}:
        return "W"
    if s in {"L", "LOSS", "LOSE", "LOST"}:
        return "L"
    if s in {"D", "DRAW"}:
        return "D"
    if s in {"NC", "NO CONTEST", "NO-CONTEST", "N/C"}:
        return "NC"

    if "NO CONTEST" in s:
        return "NC"
    if "DRAW" in s:
        return "D"
    if s.startswith("W"):
        return "W"
    if s.startswith("L"):
        return "L"
    return None

def _fallback_outcome_from_fights(fight_id: int, fid: int):
    try:
        frow = completed_fights[completed_fights["Fight_ID"] == fight_id]
        if len(frow) == 0:
            return None
        r = frow.iloc[0]

        status = str(r.get("Status", "")).strip().upper().replace("_", " ")
        if "NO CONTEST" in status or status == "NC":
            return "NC"
        if "DRAW" in status:
            return "D"

        winner = r.get("Winner_Fighter_ID")
        if pd.isna(winner):
            return None
        return "W" if int(winner) == int(fid) else "L"
    except Exception:
        return None

def record_by_id(fid: int, last_n=None):
    rows = _rows_for(fid, last_n)
    if len(rows) == 0:
        return f"No completed fights found for {fighter_name(fid)}."

    rows_u = rows.drop_duplicates(subset=["Fight_ID"]).copy()

    w = l = d = nc = 0
    unknown = 0

    for _, rr in rows_u.iterrows():
        fight_id = int(rr.get("Fight_ID")) if pd.notna(rr.get("Fight_ID")) else None
        tok = _norm_result_token(rr.get("Result", None))
        if tok is None and fight_id is not None:
            tok = _fallback_outcome_from_fights(fight_id, fid)

        if tok == "W":
            w += 1
        elif tok == "L":
            l += 1
        elif tok == "D":
            d += 1
        elif tok == "NC":
            nc += 1
        else:
            unknown += 1

    total = int(len(rows_u))

    out = [
        f"RECORD — {fighter_name(fid)}",
        f"W-L-D (NC): {w}-{l}-{d} (NC: {nc})",
        f"Total UFC fights: {total}",
    ]
    if unknown:
        out.append(f"Note: {unknown} fight(s) had unclear outcome data.")
    if last_n:
        out.append(f"Window: last {int(last_n)} completed fights")
    return "\n".join(out)


def get_age_by_id(fid: int):
    r = fighter_row(fid)
    if r is None:
        return "Fighter not found."
    if "Date_of_Birth" not in fighters.columns:
        return "Your Fighters.csv is missing Date_of_Birth."
    dob = _parse_date(r["Date_of_Birth"])
    if pd.isna(dob):
        return f"I can't read {fighter_name(fid)}'s Date_of_Birth. Fix the date format in Fighters.csv."
    today = date.today()
    born = dob.date()
    years = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    return f"{fighter_name(fid)} is {years} years old (DOB: {born})."

def _rows_for(fid: int, last_n=None):
    rows = _get_completed_rows_for_fighter(fid)
    rows = _last_n(rows, last_n)
    return rows

def sig_acc_by_id(fid: int, last_n=None):
    rows = _rows_for(fid, last_n)
    if len(rows) == 0: return f"No completed fights found for {fighter_name(fid)}."
    m = _metrics_from_rows(rows)
    return f"{fighter_name(fid)}'s significant strike accuracy is {_pct(m.get('sig_acc'))}" + (f" (last {last_n})." if last_n else ".")

def td_acc_by_id(fid: int, last_n=None):
    rows = _rows_for(fid, last_n)
    if len(rows) == 0: return f"No completed fights found for {fighter_name(fid)}."
    m = _metrics_from_rows(rows)
    return f"{fighter_name(fid)}'s takedown accuracy is {_pct(m.get('td_acc'))}" + (f" (last {last_n})." if last_n else ".")

def sig_pm_by_id(fid: int, last_n=None):
    rows = _rows_for(fid, last_n)
    if len(rows) == 0: return f"No completed fights found for {fighter_name(fid)}."
    m = _metrics_from_rows(rows)
    return f"{fighter_name(fid)} lands {_format_num(m.get('sig_pm'))} significant strikes per minute" + (f" (last {last_n})." if last_n else ".")

def td_pm_by_id(fid: int, last_n=None):
    rows = _rows_for(fid, last_n)
    if len(rows) == 0: return f"No completed fights found for {fighter_name(fid)}."
    m = _metrics_from_rows(rows)
    return f"{fighter_name(fid)} lands {_format_num(m.get('td_pm'))} takedowns per minute" + (f" (last {last_n})." if last_n else ".")

def absorbed_sig_pm_by_id(fid: int, last_n=None):
    rows = _rows_for(fid, last_n)
    if len(rows) == 0: return f"No completed fights found for {fighter_name(fid)}."
    m = _metrics_from_rows(rows)
    return f"{fighter_name(fid)} absorbs {_format_num(m.get('sig_abs_pm'))} significant strikes per minute" + (f" (last {last_n})." if last_n else ".")

def avg_sig_per_fight_by_id(fid: int, last_n=None):
    rows = _rows_for(fid, last_n)
    if len(rows) == 0: return f"No completed fights found for {fighter_name(fid)}."
    m = _metrics_from_rows(rows)
    return f"{fighter_name(fid)} averages {_format_num(m.get('avg_sig_per_fight'))} significant strikes landed per fight" + (f" (last {last_n})." if last_n else ".")

def avg_td_per_fight_by_id(fid: int, last_n=None):
    rows = _rows_for(fid, last_n)
    if len(rows) == 0: return f"No completed fights found for {fighter_name(fid)}."
    m = _metrics_from_rows(rows)
    return f"{fighter_name(fid)} averages {_format_num(m.get('avg_td_per_fight'))} takedowns per fight" + (f" (last {last_n})." if last_n else ".")

def avg_control_per_fight_by_id(fid: int, last_n=None):
    rows = _rows_for(fid, last_n)
    if len(rows) == 0: return f"No completed fights found for {fighter_name(fid)}."
    m = _metrics_from_rows(rows)
    return f"{fighter_name(fid)} averages {_format_seconds(m.get('avg_ctrl_sec_per_fight'))} of control time per fight" + (f" (last {last_n})." if last_n else ".")

def avg_fight_time_by_id(fid: int, last_n=None):
    rows = _rows_for(fid, last_n)
    if len(rows) == 0: return f"No completed fights found for {fighter_name(fid)}."
    m = _metrics_from_rows(rows)
    return f"{fighter_name(fid)}'s average fight time is {_format_seconds(m.get('avg_time_sec'))}" + (f" (last {last_n})." if last_n else ".")

def defense_by_id(fid: int, last_n=None):
    rows = _rows_for(fid, last_n)
    if len(rows) == 0: return f"No completed fights found for {fighter_name(fid)}."
    m = _metrics_from_rows(rows)
    return "\n".join([
        f"DEFENSE — {fighter_name(fid)}" + (f" (last {last_n})" if last_n else ""),
        f"{fighter_name(fid)} absorbs {_format_num(m.get('sig_abs_pm'))} sig strikes/min.",
        f"{fighter_name(fid)} allows {_format_num(m.get('td_allowed_p15'))} takedowns/15.",
        f"{fighter_name(fid)} allows {_format_seconds(m.get('ctrl_allowed_p15_sec'))} control/15.",
    ])

# ============================================================
# Prediction (with weight class sanity + size bias)
# ============================================================
# ============================================================
# Prediction v2 modifiers (styles, stance history, physicals, finishes)
# ============================================================
RULE_VERSION = 2

TD_HIGH_AVG_PER_FIGHT = 2.0      # your rule: >2 avg TD per fight is "high"
TDD_POOR_THRESHOLD = 0.60        # your rule: <60% is poor
REACH_EDGE_PER_INCH = 0.015      # small, never dominates
HEIGHT_EDGE_PER_INCH = 0.008     # tiny
AGE_EDGE_PER_YEAR = 0.010        # small
PRIME_AGE_LOW = 26
PRIME_AGE_HIGH = 33

def _safe_float(x):
    try:
        if pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def _get_inches(val):
    # Accepts numeric or strings like "72" or "72.0"
    x = pd.to_numeric(val, errors="coerce")
    return float(x) if pd.notna(x) else float("nan")

def _age_years(fid: int):
    r = fighter_row(fid)
    if r is None or "Date_of_Birth" not in fighters.columns:
        return float("nan")
    dob = _parse_date(r.get("Date_of_Birth"))
    if pd.isna(dob):
        return float("nan")
    today = date.today()
    born = dob.date()
    years = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    return float(years)

def _tdd_value(fid: int, rows: pd.DataFrame = None):
    # TDD = 1 - (opponent TD landed / opponent TD attempts)
    if rows is None:
        rows = _get_completed_rows_for_fighter(fid)
    if len(rows) == 0:
        return float("nan")
    att = rows.get("Opp_TD_Att", pd.Series(dtype=float)).sum(skipna=True)
    land = rows.get("Opp_TD_Landed", pd.Series(dtype=float)).sum(skipna=True)
    if pd.isna(att) or att == 0:
        return float("nan")
    tdd = 1.0 - (land / att)
    try:
        return max(0.0, min(1.0, float(tdd)))
    except Exception:
        return float("nan")

def _is_high_level_wrestler(metrics: dict):
    # You specified avg takedowns per fight > 2 is high.
    avg_td = _safe_float(metrics.get("avg_td_per_fight", float("nan")))
    return pd.notna(avg_td) and avg_td > TD_HIGH_AVG_PER_FIGHT

def _is_primary_striker(metrics: dict):
    # Simple heuristic: low wrestling volume, some striking output
    td_p15 = _safe_float(metrics.get("td_p15", float("nan")))
    sig_pm = _safe_float(metrics.get("sig_pm", float("nan")))
    return (pd.notna(td_p15) and td_p15 < 1.0) and (pd.notna(sig_pm) and sig_pm >= 2.2)

_FINISH_CACHE = {}

def _finish_history(fid: int):
    # Returns counts/rates from completed fights (Winner + Method)
    if fid in _FINISH_CACHE:
        return _FINISH_CACHE[fid]

    cf = completed_fights.copy()
    if "Method" not in cf.columns or "Winner_Fighter_ID" not in cf.columns:
        out = {"wins": 0, "losses": 0, "ko_wins": 0, "sub_wins": 0, "dec_wins": 0, "ko_losses": 0, "sub_losses": 0}
        _FINISH_CACHE[fid] = out
        return out

    # fights involving fighter
    involved = cf[(cf.get("Fighter_A_ID") == fid) | (cf.get("Fighter_B_ID") == fid)].copy()
    if len(involved) == 0:
        out = {"wins": 0, "losses": 0, "ko_wins": 0, "sub_wins": 0, "dec_wins": 0, "ko_losses": 0, "sub_losses": 0}
        _FINISH_CACHE[fid] = out
        return out

    wins = 0
    losses = 0
    ko_wins = 0
    sub_wins = 0
    dec_wins = 0
    ko_losses = 0
    sub_losses = 0

    for _, row in involved.iterrows():
        winner = pd.to_numeric(row.get("Winner_Fighter_ID"), errors="coerce")
        method = str(row.get("Method", "")).upper()

        is_win = pd.notna(winner) and int(winner) == int(fid)
        is_loss = pd.notna(winner) and int(winner) != int(fid)

        if is_win:
            wins += 1
            if "KO" in method or "TKO" in method:
                ko_wins += 1
            elif "SUB" in method:
                sub_wins += 1
            else:
                dec_wins += 1
        elif is_loss:
            losses += 1
            if "KO" in method or "TKO" in method:
                ko_losses += 1
            elif "SUB" in method:
                sub_losses += 1

    out = {"wins": wins, "losses": losses, "ko_wins": ko_wins, "sub_wins": sub_wins, "dec_wins": dec_wins, "ko_losses": ko_losses, "sub_losses": sub_losses}
    _FINISH_CACHE[fid] = out
    return out

def _stance_edge(a_id: int, b_id: int):
    # Small edge based on A’s historical results vs B’s stance.
    a = fighter_row(a_id)
    b = fighter_row(b_id)
    if a is None or b is None or "Stance" not in fighters.columns:
        return 0.0, "stance unknown"

    b_stance = _norm_text(b.get("Stance", ""))
    if b_stance == "":
        return 0.0, "stance unknown"

    rowsA = _get_completed_rows_for_fighter(a_id)
    if len(rowsA) == 0 or "Result" not in rowsA.columns:
        return 0.0, "no stance history"

    # Filter A fights where opponent stance matches B stance
    opp_ids = rowsA.get("Opponent_ID", pd.Series(dtype=float))
    opp_ids = opp_ids.dropna().astype(int).tolist()
    if not opp_ids:
        return 0.0, "no stance history"

    opp_rows = fighters.set_index("Fighter_ID", drop=False)
    mask = []
    for oid in opp_ids:
        if oid in opp_rows.index:
            st = _norm_text(opp_rows.loc[oid].get("Stance", ""))
            mask.append(st == b_stance)
        else:
            mask.append(False)

    # Align mask with rowsA order
    try:
        rowsA2 = rowsA.copy()
        rowsA2["__mask"] = mask[:len(rowsA2)]
        sub = rowsA2[rowsA2["__mask"] == True]
    except Exception:
        return 0.0, "no stance history"

    if len(sub) < 2:
        return 0.0, "insufficient stance sample"

    res = sub["Result"].astype(str).str.upper().str.strip()
    w = float((res == "W").sum())
    l = float((res == "L").sum())
    total = w + l
    if total == 0:
        return 0.0, "insufficient stance sample"

    win_rate = w / total
    # Convert to small edge around 0: 50% => 0
    edge = (win_rate - 0.5) * 0.30  # max about +/-0.15
    note = f"A winrate vs {b_stance}: {win_rate*100:.0f}% (n={int(total)})"
    return float(edge), note

def _physical_edge_striker_vs_striker(a_id: int, b_id: int):
    ra = fighter_row(a_id)
    rb = fighter_row(b_id)
    if ra is None or rb is None:
        return 0.0, "phys unknown"

    reachA = _get_inches(ra.get("Reach", float("nan")))
    reachB = _get_inches(rb.get("Reach", float("nan")))
    heightA = _get_inches(ra.get("Height", float("nan")))
    heightB = _get_inches(rb.get("Height", float("nan")))

    ageA = _age_years(a_id)
    ageB = _age_years(b_id)

    edge = 0.0
    notes = []

    if pd.notna(reachA) and pd.notna(reachB):
        d = reachA - reachB
        edge += REACH_EDGE_PER_INCH * d
        notes.append(f"reach diff {d:.0f}in")

    if pd.notna(heightA) and pd.notna(heightB):
        d = heightA - heightB
        edge += HEIGHT_EDGE_PER_INCH * d
        notes.append(f"height diff {d:.0f}in")

    # Prime-age nudge: prefer closer to PRIME window, but small
    if pd.notna(ageA) and pd.notna(ageB):
        def prime_score(age):
            if age < PRIME_AGE_LOW:
                return -(PRIME_AGE_LOW - age)
            if age > PRIME_AGE_HIGH:
                return -(age - PRIME_AGE_HIGH)
            return 0.0
        d = prime_score(ageA) - prime_score(ageB)
        edge += AGE_EDGE_PER_YEAR * d
        notes.append("prime-age edge")

    return float(edge), (", ".join(notes) if notes else "phys n/a")


def _expected_rate(off_val, opp_allow_val):
    if pd.isna(off_val) and pd.isna(opp_allow_val):
        return float("nan")
    if pd.isna(off_val):
        return opp_allow_val
    if pd.isna(opp_allow_val):
        return off_val
    return 0.5 * off_val + 0.5 * opp_allow_val

def _get_mode_metrics(fid: int, last_n_override):
    if last_n_override:
        return _get_lastn_metrics(fid, last_n_override), f"Override last {last_n_override}"
    b, _, _ = _get_blended_metrics(fid)
    return b, f"Default all fights (last {RECENT_N_DEFAULT} weighted)"

def _fighters_same_wc_or_upcoming(a_id: int, b_id: int):
    # If they have an upcoming scheduled fight together, allow regardless.
    oppA, rowA = _upcoming_opponent(a_id)
    if rowA is not None and oppA == b_id:
        return True, "upcoming"

    # Otherwise compare their "current" fighter Weight_Class
    wa = _wc_rank(_wc_label(a_id, fighters)) if _wc_label(a_id, fighters) else None
    wb = _wc_rank(_wc_label(b_id, fighters)) if _wc_label(b_id, fighters) else None
    if wa is None or wb is None:
        return True, "unknown"
    return (wa == wb), "different"

def _size_edge_from_wc(a_id: int, b_id: int):
    # Positive means A is heavier class than B
    wca = _wc_rank(_wc_label(a_id, fighters) or "")
    wcb = _wc_rank(_wc_label(b_id, fighters) or "")
    if wca is None or wcb is None:
        return 0.0, "Unknown weight classes"
    diff = wca - wcb
    return float(diff), f"{_norm_wc(_wc_label(a_id, fighters))} vs {_norm_wc(_wc_label(b_id, fighters))} (diff {diff})"
# ---------------------------
# Safe additional modifiers (inactivity, opponent quality, mutual opponents, weight class history)
# ---------------------------

def _clamp(x, lo, hi):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return 0.0

def _fight_date_from_row(row):
    """
    Try to extract a fight date from common column names.
    Returns pandas Timestamp or NaT.
    """
    for col in ["Date", "Fight_Date", "Event_Date", "event_date", "date"]:
        if col in row.index:
            dt = pd.to_datetime(row.get(col), errors="coerce", utc=False)
            if pd.notna(dt):
                return dt
    return pd.NaT

def _last_fight_date(fid: int, rows: pd.DataFrame = None):
    """
    Uses completed rows for fighter and returns latest date as Timestamp.
    If no date columns exist, returns NaT.
    """
    if rows is None:
        rows = _get_completed_rows_for_fighter(fid)
    if rows is None or len(rows) == 0:
        return pd.NaT

    # Find best date column present
    date_cols = [c for c in ["Date", "Fight_Date", "Event_Date", "event_date", "date"] if c in rows.columns]
    if not date_cols:
        return pd.NaT

    dt = pd.to_datetime(rows[date_cols[0]], errors="coerce")
    if dt.isna().all():
        # try other date cols
        for c in date_cols[1:]:
            dt2 = pd.to_datetime(rows[c], errors="coerce")
            if not dt2.isna().all():
                dt = dt2
                break

    if dt.isna().all():
        return pd.NaT

    return dt.max()

def _days_since_last_fight(fid: int, rows: pd.DataFrame = None):
    last_dt = _last_fight_date(fid, rows)
    if pd.isna(last_dt):
        return float("nan")
    # compare to "today"
    today_dt = pd.to_datetime(date.today())
    try:
        return float((today_dt - last_dt).days)
    except Exception:
        return float("nan")

def _inactivity_edge(a_id: int, b_id: int, A_rows: pd.DataFrame, B_rows: pd.DataFrame):
    """
    Positive edge favors A (A is more active / less layoff than B).
    Very small, capped.
    """
    da = _days_since_last_fight(a_id, A_rows)
    db = _days_since_last_fight(b_id, B_rows)
    if pd.isna(da) or pd.isna(db):
        return 0.0, "inactivity n/a"

    # If someone is very inactive, apply small penalty
    # Baseline: differences matter after ~90 days.
    diff_days = db - da  # positive => B more inactive => favors A

    # Convert to years-ish scale, small multiplier
    edge = 0.08 * (diff_days / 365.0)

    # Extra penalty if someone has a huge layoff (over 18 months)
    if da >= 540:
        edge -= 0.05
    if db >= 540:
        edge += 0.05

    edge = _clamp(edge, -0.20, 0.20)
    note = f"days since last: A={int(da)}, B={int(db)}"
    return edge, note

def _opponent_ids_from_rows(rows: pd.DataFrame):
    if rows is None or len(rows) == 0:
        return []
    if "Opponent_ID" not in rows.columns:
        return []
    opp = pd.to_numeric(rows["Opponent_ID"], errors="coerce").dropna()
    return opp.astype(int).tolist()

def _simple_win_loss_from_rows(rows: pd.DataFrame):
    if rows is None or len(rows) == 0 or "Result" not in rows.columns:
        return 0, 0
    res = rows["Result"].astype(str).str.upper().str.strip()
    w = int((res == "W").sum())
    l = int((res == "L").sum())
    return w, l

def _opponent_winrate(fid: int):
    """
    Approx opponent quality proxy: opponent UFC win rate from completed_fights.
    Uses completed_fights Winner_Fighter_ID and Fighter_A_ID/B_ID.
    """
    cf = completed_fights
    needed = {"Winner_Fighter_ID", "Fighter_A_ID", "Fighter_B_ID"}
    if cf is None or any(c not in cf.columns for c in needed):
        return float("nan")

    involved = cf[(cf["Fighter_A_ID"] == fid) | (cf["Fighter_B_ID"] == fid)]
    if len(involved) == 0:
        return float("nan")

    wins = 0
    losses = 0
    for _, r in involved.iterrows():
        winner = pd.to_numeric(r.get("Winner_Fighter_ID"), errors="coerce")
        if pd.isna(winner):
            continue
        if int(winner) == int(fid):
            wins += 1
        else:
            losses += 1

    total = wins + losses
    if total == 0:
        return float("nan")
    return wins / total

def _strength_of_schedule_score(fid: int, rows: pd.DataFrame = None):
    """
    Strength of schedule proxy:
    Average opponent UFC winrate for opponents faced (simple, safe).
    """
    if rows is None:
        rows = _get_completed_rows_for_fighter(fid)
    opp_ids = _opponent_ids_from_rows(rows)
    if not opp_ids:
        return float("nan"), "sos n/a"

    wrs = []
    for oid in opp_ids:
        wr = _opponent_winrate(oid)
        if pd.notna(wr):
            wrs.append(wr)

    if len(wrs) < 3:
        return float("nan"), "sos insufficient"

    score = float(sum(wrs) / len(wrs))  # 0..1
    return score, f"avg opp winrate {score*100:.0f}% (n={len(wrs)})"

def _sos_edge(a_id: int, b_id: int, A_rows: pd.DataFrame, B_rows: pd.DataFrame):
    """
    Positive favors A if A faced tougher average opponents.
    Small, capped.
    """
    sa, noteA = _strength_of_schedule_score(a_id, A_rows)
    sb, noteB = _strength_of_schedule_score(b_id, B_rows)
    if pd.isna(sa) or pd.isna(sb):
        return 0.0, "sos n/a"

    diff = sa - sb  # 0..1 scale
    edge = _clamp(diff * 0.25, -0.25, 0.25)
    note = f"A {noteA}; B {noteB}"
    return edge, note

def _mutual_opponents_edge(a_id: int, b_id: int, A_rows: pd.DataFrame, B_rows: pd.DataFrame):
    """
    Compare how A and B did vs shared opponents (by Opponent_ID).
    Positive favors A.
    """
    if A_rows is None or B_rows is None:
        return 0.0, "mutual n/a"
    if "Opponent_ID" not in A_rows.columns or "Opponent_ID" not in B_rows.columns:
        return 0.0, "mutual n/a"

    A_map = {}
    for _, r in A_rows.iterrows():
        oid = pd.to_numeric(r.get("Opponent_ID"), errors="coerce")
        if pd.isna(oid):
            continue
        A_map[int(oid)] = str(r.get("Result", "")).upper().strip()

    B_map = {}
    for _, r in B_rows.iterrows():
        oid = pd.to_numeric(r.get("Opponent_ID"), errors="coerce")
        if pd.isna(oid):
            continue
        B_map[int(oid)] = str(r.get("Result", "")).upper().strip()

    shared = sorted(set(A_map.keys()) & set(B_map.keys()))
    if len(shared) < 2:
        return 0.0, "mutual insufficient"

    def winrate(map_):
        w = 0
        l = 0
        for oid in shared:
            res = map_.get(oid, "")
            if res == "W":
                w += 1
            elif res == "L":
                l += 1
        tot = w + l
        return (w / tot) if tot else float("nan")

    wa = winrate(A_map)
    wb = winrate(B_map)
    if pd.isna(wa) or pd.isna(wb):
        return 0.0, "mutual n/a"

    edge = _clamp((wa - wb) * 0.15, -0.15, 0.15)
    note = f"shared opponents n={len(shared)}; A winrate {wa*100:.0f}% vs B {wb*100:.0f}%"
    return edge, note

def _row_weight_class_label(row):
    """
    Try to pull a weight-class label from a fight row (not fighters table).
    """
    for col in ["Weight_Class", "WeightClass", "weight_class", "Division", "division"]:
        if col in row.index:
            return str(row.get(col) or "").strip()
    return ""

def _recent_weight_classes(fid: int, rows: pd.DataFrame, last_n: int = 5):
    if rows is None or len(rows) == 0:
        return []
    # Use last N rows in time order if possible
    r = rows.copy()
    # If there is a date column, sort by it
    date_cols = [c for c in ["Date", "Fight_Date", "Event_Date", "event_date", "date"] if c in r.columns]
    if date_cols:
        r["__dt"] = pd.to_datetime(r[date_cols[0]], errors="coerce")
        r = r.sort_values("__dt")
    r = r.tail(last_n)

    wcs = []
    for _, row in r.iterrows():
        wc = _row_weight_class_label(row)
        if wc:
            wcs.append(_norm_wc(wc))
    return wcs

def _weight_class_history_edge(a_id: int, b_id: int, A_rows: pd.DataFrame, B_rows: pd.DataFrame, upcoming_wc_label: str = ""):
    """
    Two small ideas:
    1) Stability: fewer recent weight-class switches = small advantage
    2) Familiarity: if last fight WC matches upcoming, small advantage
    """
    # If we don't have per-fight WCs, safely no-op
    A_wcs = _recent_weight_classes(a_id, A_rows, last_n=5)
    B_wcs = _recent_weight_classes(b_id, B_rows, last_n=5)
    if not A_wcs and not B_wcs:
        return 0.0, "wc history n/a"

    # Stability score: unique count (more unique => less stable)
    A_unique = len(set(A_wcs)) if A_wcs else 99
    B_unique = len(set(B_wcs)) if B_wcs else 99
    # Favor the more stable one (lower unique count)
    stability_edge = _clamp((B_unique - A_unique) * 0.05, -0.10, 0.10)

    # Familiarity: did they fight at upcoming class last time?
    familiarity_edge = 0.0
    note_fam = ""
    if upcoming_wc_label:
        up = _norm_wc(upcoming_wc_label)
        A_last = A_wcs[-1] if A_wcs else ""
        B_last = B_wcs[-1] if B_wcs else ""
        if A_last and B_last:
            if A_last == up and B_last != up:
                familiarity_edge += 0.06
                note_fam = "A last fight at upcoming WC"
            elif B_last == up and A_last != up:
                familiarity_edge -= 0.06
                note_fam = "B last fight at upcoming WC"

    edge = _clamp(stability_edge + familiarity_edge, -0.15, 0.15)
    note = f"A recent WCs={A_wcs or 'n/a'}; B recent WCs={B_wcs or 'n/a'}"
    if note_fam:
        note += f"; {note_fam}"
    return edge, note


def predict(a_id: int, b_id: int, last_n_override=None):
    key = _cache_key("predict_v2", *sorted([a_id, b_id]), last_n_override, RECENT_N_DEFAULT, RECENT_WEIGHT_DEFAULT)

    if key in PRED_CACHE:
        return PRED_CACHE[key]

    allowed, reason = _fighters_same_wc_or_upcoming(a_id, b_id)
    if not allowed:
        # Hard stop: nonsense cross-class
        # But still: per your request, it should NOT pick the smaller fighter.
        diff, wc_note = _size_edge_from_wc(a_id, b_id)
        heavier = a_id if diff > 0 else b_id
        out = "\n".join([
            f"PREDICTION — {fighter_name(a_id)} vs {fighter_name(b_id)}",
            f"NOTE: These fighters are in different weight classes ({wc_note}).",
            f"Cross-division predictions aren’t meaningful, so I’m defaulting to the heavier weight class.",
            f"Pick: {fighter_name(heavier)} (size/class advantage)",
        ])
        PRED_CACHE[key] = out
        _save_cache()
        return out

    # Determine rounds if upcoming fight exists between them
    rounds_scheduled = 3
    oppA, rowA = _upcoming_opponent(a_id)
    if rowA is not None and oppA == b_id and "Rounds_Scheduled" in rowA.index:
        rs = pd.to_numeric(rowA.get("Rounds_Scheduled"), errors="coerce")
        if pd.notna(rs) and int(rs) in (3, 5):
            rounds_scheduled = int(rs)

    mode = f"Override: last {last_n_override}" if last_n_override else f"Default: all fights (last {RECENT_N_DEFAULT} weighted)"

    A, _ = _get_mode_metrics(a_id, last_n_override)
    B, _ = _get_mode_metrics(b_id, last_n_override)
    score = 0.0
    contribs = []

    # --- Style/physics/stance modifiers (deterministic) ---
    # Determine "wrestler" vs "striker"
    A_wrestler = _is_high_level_wrestler(A)
    B_wrestler = _is_high_level_wrestler(B)
    A_striker = _is_primary_striker(A)
    B_striker = _is_primary_striker(B)

    # TDD values from same sample used for prediction mode
    if last_n_override:
        A_rows = _last_n(_get_completed_rows_for_fighter(a_id), last_n_override)
        B_rows = _last_n(_get_completed_rows_for_fighter(b_id), last_n_override)
    else:
        A_rows = _get_completed_rows_for_fighter(a_id)
        B_rows = _get_completed_rows_for_fighter(b_id)
    # --- Additional safe modifiers (small, capped) ---
    # Upcoming weight class (if available from upcoming row)
    upcoming_wc = ""
    try:
        # if rowA exists for upcoming matchup
        if rowA is not None:
            # try common columns
            if "Weight_Class" in rowA.index:
                upcoming_wc = str(rowA.get("Weight_Class") or "").strip()
            elif "WeightClass" in rowA.index:
                upcoming_wc = str(rowA.get("WeightClass") or "").strip()
    except Exception:
        upcoming_wc = ""

    # Inactivity / layoffs
    inact_edge, inact_note = _inactivity_edge(a_id, b_id, A_rows, B_rows)
    if inact_edge != 0.0:
        score += inact_edge
        contribs.append((abs(inact_edge), inact_edge, f"Inactivity ({inact_note})", 0.0, 0.0))

    # Strength of schedule (opponent quality proxy)
    sos_edge, sos_note = _sos_edge(a_id, b_id, A_rows, B_rows)
    if sos_edge != 0.0:
        score += sos_edge
        contribs.append((abs(sos_edge), sos_edge, f"Opponent quality ({sos_note})", 0.0, 0.0))

    # Mutual opponents
    mutual_edge, mutual_note = _mutual_opponents_edge(a_id, b_id, A_rows, B_rows)
    if mutual_edge != 0.0:
        score += mutual_edge
        contribs.append((abs(mutual_edge), mutual_edge, f"Mutual opponents ({mutual_note})", 0.0, 0.0))

    # Multi-weight class history (stability + familiarity)
    wc_hist_edge, wc_hist_note = _weight_class_history_edge(a_id, b_id, A_rows, B_rows, upcoming_wc_label=upcoming_wc)
    if wc_hist_edge != 0.0:
        score += wc_hist_edge
        contribs.append((abs(wc_hist_edge), wc_hist_edge, f"Weight class history ({wc_hist_note})", 0.0, 0.0))

    A_tdd = _tdd_value(a_id, A_rows)
    B_tdd = _tdd_value(b_id, B_rows)

    # Stance edge (small)
    stance_edge_A, stance_note_A = _stance_edge(a_id, b_id)
    stance_edge_B, stance_note_B = _stance_edge(b_id, a_id)

    # Physical edge only if striker vs striker
    phys_edge = 0.0
    phys_note = "n/a"
    if A_striker and B_striker:
        phys_edge, phys_note = _physical_edge_striker_vs_striker(a_id, b_id)


    A_exp_sigpm = _expected_rate(A.get("sig_pm"), B.get("sig_abs_pm"))
    B_exp_sigpm = _expected_rate(B.get("sig_pm"), A.get("sig_abs_pm"))

    A_exp_tdp15 = _expected_rate(A.get("td_p15"), B.get("td_allowed_p15"))
    B_exp_tdp15 = _expected_rate(B.get("td_p15"), A.get("td_allowed_p15"))

    A_exp_ctrlp15 = _expected_rate(A.get("ctrl_p15_sec"), B.get("ctrl_allowed_p15_sec"))
    B_exp_ctrlp15 = _expected_rate(B.get("ctrl_p15_sec"), A.get("ctrl_allowed_p15_sec"))

    A_exp_kdpm = _expected_rate(A.get("kd_pm"), B.get("kd_pm"))  # keep simple (you can refine later)
    B_exp_kdpm = _expected_rate(B.get("kd_pm"), A.get("kd_pm"))

    # Weight-class size edge (only applied if ranks known)
    size_diff, size_note = _size_edge_from_wc(a_id, b_id)
    # If same WC, diff = 0. If unknown, diff=0.
    # Convert diff to a small edge term:
    # +1 class ~= +0.25 normalized score (enough to prevent flyweight > HW in close calls)
    size_edge = 0.25 * size_diff

    components = [
        ("Expected Sig / min", A_exp_sigpm, B_exp_sigpm, 2.2),
        ("Expected TD / 15", A_exp_tdp15, B_exp_tdp15, 1.6),
        ("Expected Control / 15", A_exp_ctrlp15, B_exp_ctrlp15, 1.2),
        ("Sig accuracy", A.get("sig_acc", float("nan")), B.get("sig_acc", float("nan")), 0.8),
        ("TD accuracy", A.get("td_acc", float("nan")), B.get("td_acc", float("nan")), 0.7),
        ("Sig absorbed / min (lower better)", -A.get("sig_abs_pm", float("nan")), -B.get("sig_abs_pm", float("nan")), 0.8),
        ("TD allowed / 15 (lower better)", -A.get("td_allowed_p15", float("nan")), -B.get("td_allowed_p15", float("nan")), 0.8),
    ]
    # If both are high-level wrestlers, fights often become striking matches:
    # downweight TD/control, upweight striking
    if A_wrestler and B_wrestler:
        new_components = []
        for label, a_val, b_val, w in components:
            if "TD" in label or "Control" in label:
                new_components.append((label, a_val, b_val, w * 0.55))
            elif "Sig" in label or "accuracy" in label or "absorbed" in label.lower():
                new_components.append((label, a_val, b_val, w * 1.15))
            else:
                new_components.append((label, a_val, b_val, w))
        components = new_components

    
    # Extra matchup components (style logic)
    # Wrestler vs poor TDD: boost wrestler edge
    if B_wrestler and pd.notna(A_tdd) and A_tdd < TDD_POOR_THRESHOLD:
        contribs.append((0.50, -0.50, "Wrestling vs poor TDD", 0.0, 0.0))  # favors B (negative score)
        score -= 0.50

    if A_wrestler and pd.notna(B_tdd) and B_tdd < TDD_POOR_THRESHOLD:
        contribs.append((0.50, 0.50, "Wrestling vs poor TDD", 0.0, 0.0))   # favors A (positive score)
        score += 0.50

    # Striker vs striker: apply small physical edge
    if phys_edge != 0.0:
        contribs.append((abs(phys_edge), phys_edge, f"Physical edge ({phys_note})", phys_edge, 0.0))
        score += phys_edge

    # Stance-history edge (very small)
    if stance_edge_A != 0.0:
        contribs.append((abs(stance_edge_A), stance_edge_A, f"Stance history ({stance_note_A})", stance_edge_A, 0.0))
        score += stance_edge_A
    if stance_edge_B != 0.0:
    # If B has stance advantage, that should PUSH score toward B (negative for A)
        contribs.append((abs(stance_edge_B), -stance_edge_B, f"Stance history (B view: {stance_note_B})", -stance_edge_B, 0.0))
        score -= stance_edge_B



    for label, a_val, b_val, w in components:
        edge = _norm_edge(a_val, b_val)
        if pd.isna(edge):
            continue
        c = w * edge
        score += c
        contribs.append((abs(c), c, label, a_val, b_val))

    # Apply size edge last (deterministic, never flips)
    if size_edge != 0:
        score += size_edge
        contribs.append((abs(size_edge), size_edge, "Weight class edge", size_edge, 0.0))

    winner_is_A = score > 0
    winner = fighter_name(a_id) if winner_is_A else fighter_name(b_id)
    tier = _tier_from_strength(abs(score))

    sched_min = 5.0 * rounds_scheduled
    raw_exp_min = _expected_fight_minutes(A, B, rounds_scheduled)
    snap_exp_min = _snap_expected_minutes(raw_exp_min, rounds_scheduled)
    display_time_min = snap_exp_min
    # ---------------- Final method pick ----------------
    if snap_exp_min >= 0.85 * sched_min:

        method = "Decision"
        rnd = "Decision"
        display_time_min = sched_min

    else:
        fhA = _finish_history(a_id)
        fhB = _finish_history(b_id)

        A_ko_score = (fhA["ko_wins"] + 0.25) * (fhB["ko_losses"] + 0.25)
        B_ko_score = (fhB["ko_wins"] + 0.25) * (fhA["ko_losses"] + 0.25)

        A_sub_score = (fhA["sub_wins"] + 0.25) * (fhB["sub_losses"] + 0.25)
        B_sub_score = (fhB["sub_wins"] + 0.25) * (fhA["sub_losses"] + 0.25)

        if winner_is_A:
            if fhA["ko_wins"] == 0 and fhB["ko_losses"] == 0:
                A_ko_score *= 0.35
            if fhB["ko_losses"] == 0:
                A_ko_score *= 0.55
            best_finish = "KO" if A_ko_score >= A_sub_score else "SUB"
        else:
            if fhB["ko_wins"] == 0 and fhA["ko_losses"] == 0:
                B_ko_score *= 0.35
            if fhA["ko_losses"] == 0:
                B_ko_score *= 0.55
            best_finish = "KO" if B_ko_score >= B_sub_score else "SUB"

        if best_finish == "SUB":
            method = "Submission"
            finish_min = raw_exp_min * 0.88
            display_time_min = finish_min

        else:
            method = "KO/TKO"
            finish_min = raw_exp_min * 0.78
            display_time_min = finish_min

        r = int((max(0.01, min(display_time_min, sched_min)) - 0.01) // 5.0) + 1
        r = max(1, min(r, rounds_scheduled))
        rnd = f"Round {r}"

    # ---------------- Final output ----------------
    sched_min = 5.0 * rounds_scheduled

    # For finishes, show the estimated FINISH time (the same time used to choose Round X)
    if method == "Decision":
        shown_time_min = sched_min
    else:
        # display_time_min was set earlier for KO/SUB (finish_min)
        shown_time_min = min(display_time_min, sched_min)

    out = "\n".join([
        f"PREDICTION — {fighter_name(a_id)} vs {fighter_name(b_id)}",
        f"Confidence: {tier}",
        f"Pick: {winner}",
        f"Method pick: {method} ({rnd})",
        "",
        f"Expected fight time: {shown_time_min:.2f} min (scheduled {sched_min:.0f})",
    ])

    PRED_CACHE[key] = out
    _save_cache()
    return out



# ============================================================
# English router (keeps EVERYTHING)
# ============================================================
def _parse_prop_from_text(lowc: str):
    pick = None
    if "over" in lowc:
        pick = "over"
    if "under" in lowc:
        pick = "under"

    m = re.search(r"\b(\d+(\.\d+)?)\b", lowc)
    if not m:
        return None
    line = float(m.group(1))

    stat = None
    if any(k in lowc for k in ["sig", "significant", "strikes", "strike"]):
        stat = "sig"
    elif any(k in lowc for k in ["takedown", "takedowns", "td"]):
        stat = "td"
    elif any(k in lowc for k in ["minutes", "mins", "min", "time", "duration"]):
        stat = "minutes"

    if stat is None:
        return None

    return {"pick": pick, "line": line, "stat": stat}


def parse_last_n(text: str):
    m = re.search(r"\blast\s+(\d+)\b", text.lower())
    if not m:
        return None, text
    n = int(m.group(1))
    new_text = re.sub(r"\blast\s+\d+\b", "", text, flags=re.IGNORECASE).strip()
    return n, new_text

def _strip_name_guess(lowc: str, words_to_remove):
    s = lowc
    for w in words_to_remove:
        s = re.sub(rf"\b{re.escape(w)}\b", " ", s)
    s = re.sub(r"\b(\d+(\.\d+)?)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def english_router(q: str):
    original = q.strip()
    cleaned = re.sub(r"[?.,!]", " ", original)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    lowc = cleaned.lower()
    # Gist matchup detection:
    # If the user rambles but includes exactly 2 fighter names, treat as a matchup
    # ONLY when there is some matchup cue.
    matchup_cue = any(k in lowc for k in [
        " vs ", " versus ", " v ", " or ", "who you got", "who do you got",
        "who wins", "who will win", "pick", "got", "fight", "bout", "matchup", "against"
    ])

    if matchup_cue:
        pair = _extract_two_fighters_anywhere(cleaned)
        if pair:
            a_id, b_id = pair
            return f"predict {fighter_name(a_id)} vs {fighter_name(b_id)}"

    # last N / recent
    last_n = None
    m = re.search(r"\blast\s+(\d+)\b", lowc)
    if m:
        last_n = int(m.group(1))
    elif any(w in lowc for w in ["recent", "recently", "lately", "last few"]):
        last_n = 5
    last_prefix = f"last {last_n} " if last_n else ""

    # Last N fights
    m = re.search(r"last\s+(\d+)\s+fights?", lowc)
    if m:
        n = m.group(1)
        metric = "summary"
        if any(k in lowc for k in ["sig", "significant", "strikes", "strike"]):
            metric = "sig"
        elif any(k in lowc for k in ["takedown", "takedowns", "td"]):
            metric = "td"
        elif "control" in lowc:
            metric = "control"
        elif any(k in lowc for k in ["how long", "duration", "minutes", "time"]):
            metric = "minutes"
        elif any(k in lowc for k in ["knockdown", "knockdowns", "kd"]):
            metric = "knockdowns"

        name_guess = _strip_name_guess(lowc, ["last", n, "fights", "fight", "in", "of", "his", "her", "their", "average", "avg", "what", "is", "s", "how", "many", "did", "land", "the", "sig", "significant", "strikes", "strike", "takedown", "takedowns", "td", "control", "time", "duration", "minutes", "knockdown", "knockdowns", "kd"])
        return f"lastn {n} {metric} {name_guess}".strip()

    # Props / Over-Under (natural English)
    if any(k in lowc for k in ["over", "under", "over/under", "over under", "prop", "props", "prizepicks"]):
        info = _parse_prop_from_text(lowc)
        if info:
            tmp = re.sub(r"\bover/under\b", " ", lowc)
            tmp = re.sub(r"\bover under\b", " ", tmp)
            tmp = re.sub(r"\bprizepicks\b", " ", tmp)
            tmp = re.sub(r"\bprops?\b", " ", tmp)
            tmp = re.sub(r"\bover\b", " ", tmp)
            tmp = re.sub(r"\bunder\b", " ", tmp)
            tmp = re.sub(r"\bfor\b", " ", tmp)
            tmp = re.sub(r"\bat\b", " ", tmp)
            tmp = re.sub(r"\b(\d+(\.\d+)?)\b", " ", tmp)
            tmp = re.sub(r"\b(sig|significant|strikes|strike|takedown|takedowns|td|minutes|mins|min|time|duration)\b", " ", tmp)
            tmp = re.sub(r"\s+", " ", tmp).strip()

            if " vs " in tmp:
                a, b = tmp.split(" vs ", 1)
                return f"props {a.strip()} vs {b.strip()} {info['stat']} {info['line']}"
            return f"props {tmp.strip()} {info['stat']} {info['line']}"


    # Last fight (with optional metric)
    if "last fight" in lowc:
        metric = "summary"
        if any(k in lowc for k in ["sig", "significant", "strikes", "strike"]):
            metric = "sig"
        elif any(k in lowc for k in ["takedown", "takedowns", "td"]):
            metric = "td"
        elif "control" in lowc:
            metric = "control"
        elif any(k in lowc for k in ["how long", "duration", "minutes", "time"]):
            metric = "minutes"
        elif any(k in lowc for k in ["knockdown", "knockdowns", "kd"]):
            metric = "knockdowns"

        name_guess = _strip_name_guess(lowc, ["last", "fight", "in", "of", "his", "her", "their", "stats", "stat", "what", "is", "s", "how", "many", "did", "land", "in", "the", "sig", "significant", "strikes", "strike", "takedown", "takedowns", "td", "control", "time", "duration", "minutes", "knockdown", "knockdowns", "kd", "long"])
        return f"lastfight {metric} {name_guess}".strip()


    # Record
    if any(k in lowc for k in ["record", "ufc fights", "how many fights", "total fights", "fight count"]):
        name_guess = _strip_name_guess(lowc, ["record", "ufc", "fights", "fight", "how", "many", "total", "count", "does", "have", "what", "is", "s"])
        return f"record {name_guess}".strip()
   
 # Takedown defense
    if any(k in lowc for k in ["takedown defense", "tdd", "td defense"]):
        name_guess = _strip_name_guess(lowc, ["takedown", "td", "defense", "tdd", "what", "is"])
        return f"tdd {name_guess}".strip()

    # Style tags
    if any(k in lowc for k in ["style tag", "style tags", "style"]):
        name_guess = _strip_name_guess(lowc, ["style", "tag", "tags"])
        return f"style {name_guess}".strip()

    # Recent form
    if any(k in lowc for k in ["recent form", "form", "trajectory", "streak"]):
        name_guess = _strip_name_guess(lowc, ["recent", "form", "trajectory", "streak"])
        return f"form {name_guess}".strip()

    # Data health checks
    if any(k in lowc for k in ["data health", "health check", "data check", "check data"]):
        return "health"

    # Matchup intent
    matchup_intent = any(k in lowc for k in [" vs ", " v ", " versus ", " or ", "who wins", "who will win"])
    if matchup_intent:
        tmp = re.sub(r"\bversus\b", "vs", lowc)
        tmp = re.sub(r"\bv\b", "vs", tmp)
        if " vs " in tmp:
            left, right = tmp.split(" vs ", 1)
        elif " or " in tmp:
            left, right = tmp.split(" or ", 1)
        else:
            left, right = None, None

        if left and right:
            a_res = find_fighter_id(left, allow_ask=True)
            b_res = find_fighter_id(right, allow_ask=True)

            if isinstance(a_res, tuple) and a_res[0] == "__AMBIG__":
                return _ask_clarify(a_res[1], lambda fid: f"predict {fighter_name(fid)} vs {right.strip()}", title="Multiple matches for the first fighter. ")
            if isinstance(b_res, tuple) and b_res[0] == "__AMBIG__":
                return _ask_clarify(b_res[1], lambda fid: f"predict {left.strip()} vs {fighter_name(fid)}", title="Multiple matches for the second fighter. ")

            if isinstance(a_res, int) and isinstance(b_res, int):
                return f"predict {fighter_name(a_res)} vs {fighter_name(b_res)}"

    # Average/per-fight intents
    want_avg = any(w in lowc for w in ["average", "avg", "per fight"])
    want_td = any(w in lowc for w in ["takedown", "takedowns", "td"])
    want_ctrl = "control" in lowc
    want_sig = any(w in lowc for w in ["significant", "sig", "strikes", "strike"])
    want_per_min = any(w in lowc for w in ["per minute", "per min", "pm"])

    if want_avg and want_td and ("accuracy" not in lowc):
        name_guess = _strip_name_guess(lowc, ["average","avg","per","fight","takedown","takedowns","td","number","of"])
        return f"avg td fight {name_guess}".strip()

    if want_avg and want_ctrl:
        name_guess = _strip_name_guess(lowc, ["average","avg","per","fight","control","time","number","of"])
        return f"avg ctrl fight {name_guess}".strip()

    if want_avg and want_sig and ("accuracy" not in lowc) and not want_per_min:
        name_guess = _strip_name_guess(lowc, ["average","avg","per","fight","significant","sig","strike","strikes","number","of","landed"])
        return f"avg sig fight {name_guess}".strip()

    if want_avg and any(w in lowc for w in ["fight time", "duration", "time"]):
        name_guess = _strip_name_guess(lowc, ["average","avg","fight","time","duration","how","long","is","their"])
        return f"avg time {name_guess}".strip()

    # Per-minute intents
    if want_sig and want_per_min and ("accuracy" not in lowc):
        name_guess = _strip_name_guess(lowc, ["per","minute","min","pm","significant","sig","strike","strikes","landed"])
        return f"sig pm {last_prefix}{name_guess}".strip()

    if want_td and want_per_min and ("accuracy" not in lowc):
        name_guess = _strip_name_guess(lowc, ["per","minute","min","pm","takedown","takedowns","td"])
        return f"td pm {last_prefix}{name_guess}".strip()

    if "absorbed" in lowc and want_per_min:
        name_guess = _strip_name_guess(lowc, ["absorbed","absorb","per","minute","min","pm","strikes","strike","significant","sig"])
        return f"abs sig pm {last_prefix}{name_guess}".strip()

    # Accuracy + age + defense
    want_age = ("how old" in lowc) or (" age" in f" {lowc} ")
    want_sig_acc = any(k in lowc for k in ["strike accuracy","striking accuracy","sig accuracy","significant strike accuracy"])
    want_td_acc = any(k in lowc for k in ["takedown accuracy","td accuracy"])
    want_def = any(k in lowc for k in ["defense", "allowed"])

    # Find fighter in sentence
    fid_res = find_fighter_id(cleaned, allow_ask=True)
    if isinstance(fid_res, tuple) and fid_res[0] == "__AMBIG__":
        def build(fid):
            nm = fighter_name(fid)
            if want_age:
                return f"age {nm}"
            if want_sig_acc:
                return f"sig acc {last_prefix}{nm}".strip()
            if want_td_acc:
                return f"td acc {last_prefix}{nm}".strip()
            if want_def:
                return f"defense {last_prefix}{nm}".strip()
            return "help"
        return _ask_clarify(fid_res[1], build, title="I found multiple fighters with that name. ")

    fid = fid_res if isinstance(fid_res, int) else None
    if fid is None:
        return "Which fighter are you asking about? (Type the name.)"

    if want_age:
        return f"age {fighter_name(fid)}"
    if want_sig_acc:
        return f"sig acc {last_prefix}{fighter_name(fid)}".strip()
    if want_td_acc:
        return f"td acc {last_prefix}{fighter_name(fid)}".strip()
    if want_def:
        return f"defense {last_prefix}{fighter_name(fid)}".strip()

    # If fighter found but metric unclear
    return f"I found the fighter (**{fighter_name(fid)}**), but I didn’t catch the stat. Try: 'style tags', 'recent form', 'strikes per minute', 'average takedowns per fight', etc."

# ============================================================
# Command handler
# ============================================================
HELP_TEXT = """
Natural English examples:
  Rose Namajunas average number of takedowns per fight
  Umar Nurmagomedov average control time per fight
  Max Holloway strikes per minute
  Max Holloway absorbed strikes per minute
  Umar recent form
  Umar style tags
  data health
  Derrick Lewis vs Alex Perez
  Alex Perez vs Charles Johnson

Direct commands:
  age <name>
  sig acc <name>
  td acc <name>
  defense <name>
  sig pm <name>
  abs sig pm <name>
  td pm <name>
  avg sig fight <name>
  avg td fight <name>
  avg ctrl fight <name>
  avg time <name>
  style <name>
  form <name>
  tdd <name>
  health
  predict <A> vs <B>
  props <fighter> <sig|td|minutes> <line>
  props <fighter> vs <opponent> <sig|td|minutes> <line>
""".strip()

def handle_query(q: str):
    q = q.strip()
    if q == "":
        return None


    # Pending yes/no clarification
    if STATE.get("pending"):
        ans = _norm_text(q)
        pending = STATE["pending"]
        if pending.get("kind") == "props_need_opp":
            opp_text = q.strip().strip('"').strip("'")
            opp_res = find_fighter_id(opp_text, allow_ask=True)
            if isinstance(opp_res, tuple) and opp_res[0] == "__AMBIG__":
                return _ask_clarify(
                    opp_res[1],
                    lambda fid2: f"props {fighter_name(pending['fid'])} vs {fighter_name(fid2)} {pending['stat']} {pending['line']}",
                    title="Multiple matches for the opponent. "
                )
            if not isinstance(opp_res, int):
                return "I couldn't match that opponent name. Try the full name."
            STATE["pending"] = None
            return props_over_under_by_id(
            pending["fid"],
            pending["line"],
            pending["stat"],
            opp_name=fighter_name(opp_res),
        )            



        if ans in YES_WORDS:
            cmd = pending["yes_command"]
            STATE["pending"] = None
            q = cmd
        elif ans in NO_WORDS:
            return pending["no_prompt"]
        else:
            fid = find_fighter_id(q, allow_ask=False)
            if isinstance(fid, int):
                STATE["pending"] = None
                return f"Got it: **{fighter_name(fid)}**. Now re-type your question using that name."
            return pending["question"]

    low = q.lower()
    if low in {"help", "h", "?"}:
        return HELP_TEXT
    if low in {"quit", "exit"}:
        _save_cache()
        return "__QUIT__"

       # Direct commands should bypass the English router (router is for natural language)
    direct_prefixes = (
    "age ",
    "sig acc ",
    "td acc ",
    "defense ",
    "sig pm ",
    "abs sig pm ",
    "td pm ",
    "avg sig fight ",
    "avg td fight ",
    "avg ctrl fight ",
    "avg time ",
    "style ",
    "form ",
    "props ",
    "predict ",
    "record ",
    "health",
    "tdd ",
    "lastfight ",
    "lastn ",
)


    if any(low.startswith(p) for p in direct_prefixes):
        routed = q
    else:
        routed = english_router(q)


    if STATE.get("pending"):
        return routed

    # If router returns a user-facing message
    if routed.startswith("I found the fighter") or routed.startswith("Which fighter"):
        return routed

    q = routed
    last_n, q2 = parse_last_n(q)
    low2 = q2.lower()
    # props: props <fighter> <stat> <line>  OR  props <fighter> vs <opponent> <stat> <line>
    if low2.startswith("props "):
        body = q2[len("props "):].strip()

        # Try to parse "vs"
        opp_name = None
        if re.search(r"\s+vs\s+", body, flags=re.IGNORECASE):
            left, right = re.split(r"\s+vs\s+", body, maxsplit=1, flags=re.IGNORECASE)
            fighter_part = left.strip()
            rest = right.strip()
            parts = rest.split()
            if len(parts) < 3:
                return "Usage: props <fighter> vs <opponent> <sig|td|minutes> <line>"
            # opponent name may be multiple words: everything except last 2 tokens
            stat = parts[-2]
            try:
                line = float(parts[-1])
            except ValueError:
                return "Line must be a number, e.g. 62.5"
            opp_name = " ".join(parts[:-2]).strip()
            fighter_name_text = fighter_part
        else:
            parts = body.split()
            if len(parts) < 3:
                return "Usage: props <fighter> <sig|td|minutes> <line> (or add: vs <opponent>)"
            stat = parts[-2]
            try:
                line = float(parts[-1])
            except ValueError:
                return "Line must be a number, e.g. 62.5"
            fighter_name_text = " ".join(parts[:-2]).strip()

        res = find_fighter_id(fighter_name_text, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            if opp_name:
                return _ask_clarify(res[1], lambda fid: f"props {fighter_name(fid)} vs {opp_name} {stat} {line}", title="Multiple matches for the fighter. ")
            return _ask_clarify(res[1], lambda fid: f"props {fighter_name(fid)} {stat} {line}", title="Multiple matches for the fighter. ")
        if not isinstance(res, int):
            return "Which fighter?"

        out = props_over_under_by_id(res, line=line, stat=stat, opp_name=opp_name)
        if isinstance(out, tuple) and out[0] == "__AMBIG_OPP__":
            return _ask_clarify(out[1], lambda fid2: f"props {fighter_name(res)} vs {fighter_name(fid2)} {stat} {line}", title="Multiple matches for the opponent. ")
        return out


    if low2 == "health":
        return data_health()

    # style
    if low2.startswith("style "):
        name = q2[6:].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"style {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter for style tags?"
        return style_tags(res, last_n or 5)

    # form
    if low2.startswith("form "):
        name = q2[5:].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"form {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter for recent form?"
        return recent_form(res, last_n or 5)

    # age
    if low2.startswith("age "):
        name = q2[4:].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"age {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter? Try full name."
        return get_age_by_id(res)

    # record
    if low2.startswith("record "):
        name = q2[7:].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"record {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter? Try full name."
        return record_by_id(res, last_n)
   
 # last fight
    if low2.startswith("last fight "):
        name = q2[len("last fight "):].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"last fight {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter?"
        return last_fight_stat_by_id(res, "summary")
   
 # last fight (metric)
    if low2.startswith("lastfight "):
        rest = q2[len("lastfight "):].strip()
        parts = rest.split(" ", 1)
        metric = parts[0].strip().lower() if parts else "summary"
        name = parts[1].strip() if len(parts) > 1 else ""
        if name == "":
            return "Which fighter?"
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"lastfight {metric} {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter?"
        return last_fight_stat_by_id(res, metric)

    # last N fights
    if low2.startswith("lastn "):
        rest = q2[len("lastn "):].strip()
        parts = rest.split(" ", 2)
        if len(parts) < 2:
            return "Usage: lastn <number> <fighter name>"
        try:
            n = int(parts[0])
        except ValueError:
            return "How many fights?"
        metric = parts[1].lower()
        name = parts[2].strip() if len(parts) > 2 else ""
        if name == "":
            return "Which fighter?"
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"lastn {n} {metric} {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter?"
        return last_n_fights_stat_by_id(res, n, metric)


    # sig acc
    if low2.startswith("sig acc "):
        name = q2[8:].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"sig acc {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter?"
        return sig_acc_by_id(res, last_n)

    # td acc
    if low2.startswith("td acc "):
        name = q2[7:].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"td acc {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter?"
        return td_acc_by_id(res, last_n)

    # defense
    if low2.startswith("defense "):
        name = q2[8:].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"defense {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter?"
        return defense_by_id(res, last_n)

    # takedown defense
    if low2.startswith("tdd "):
        name = q2[4:].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"tdd {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter?"
        return td_def_by_id(res, last_n)
    
        name = q2[8:].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"defense {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter?"
        return defense_by_id(res, last_n)

    # sig pm
    if low2.startswith("sig pm "):
        name = q2[7:].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"sig pm {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter?"
        return sig_pm_by_id(res, last_n)

    # abs sig pm
    if low2.startswith("abs sig pm "):
        name = q2[len("abs sig pm "):].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"abs sig pm {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter?"
        return absorbed_sig_pm_by_id(res, last_n)

    # td pm
    if low2.startswith("td pm "):
        name = q2[6:].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"td pm {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter?"
        return td_pm_by_id(res, last_n)

    # avg sig fight
    if low2.startswith("avg sig fight "):
        name = q2[len("avg sig fight "):].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"avg sig fight {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter?"
        return avg_sig_per_fight_by_id(res, last_n)

    # avg td fight
    if low2.startswith("avg td fight "):
        name = q2[len("avg td fight "):].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"avg td fight {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter?"
        return avg_td_per_fight_by_id(res, last_n)

    # avg ctrl fight
    if low2.startswith("avg ctrl fight "):
        name = q2[len("avg ctrl fight "):].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"avg ctrl fight {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter?"
        return avg_control_per_fight_by_id(res, last_n)

    # avg time
    if low2.startswith("avg time "):
        name = q2[len("avg time "):].strip()
        res = find_fighter_id(name, allow_ask=True)
        if isinstance(res, tuple) and res[0] == "__AMBIG__":
            return _ask_clarify(res[1], lambda fid: f"avg time {fighter_name(fid)}", title="Multiple matches. ")
        if not isinstance(res, int):
            return "Which fighter?"
        return avg_fight_time_by_id(res, last_n)

    # predict
    if low2.startswith("predict "):
        body = q2[8:].strip()
        if " vs " not in body.lower():
            return "Example: predict Alex Perez vs Charles Johnson"
        a, b = re.split(r"\s+vs\s+", body, flags=re.IGNORECASE)
        a_res = find_fighter_id(a.strip(), allow_ask=True)
        b_res = find_fighter_id(b.strip(), allow_ask=True)
        if isinstance(a_res, tuple) and a_res[0] == "__AMBIG__":
            return _ask_clarify(a_res[1], lambda fid: f"predict {fighter_name(fid)} vs {b.strip()}", title="Multiple matches for the first fighter. ")
        if isinstance(b_res, tuple) and b_res[0] == "__AMBIG__":
            return _ask_clarify(b_res[1], lambda fid: f"predict {a.strip()} vs {fighter_name(fid)}", title="Multiple matches for the second fighter. ")
        if not isinstance(a_res, int) or not isinstance(b_res, int):
            return "I couldn’t match one fighter — try full names."
        return predict(a_res, b_res, last_n_override=last_n)

    return "Type: help"







