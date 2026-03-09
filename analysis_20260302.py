
import os
import re
import math
import colorsys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib.colors import to_hex, to_rgb
from matplotlib.ticker import PercentFormatter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# ===== 設定 =====
INPUT_CSV = "/Users/matsuihiroshi/Dropbox/ongoingExperiment/quantitative_history/journal_trend_analysis/journal_info.xlsx"
BASE_DIR = os.path.dirname(os.path.abspath(INPUT_CSV))
OUTDIR = os.path.join(BASE_DIR, "out")
MAX_FEATURES = 30000
NGRAM_MAX = 2
LANG = "en"
SEED = 42
TOP_TERMS = 10
TREND_WINDOW = 10

CUSTOM_STOPWORDS = {
    "study", "studies", "experimental", "psychology", "physiology", "wilhelm", "stimuli",
    "psychological", "relation", "i", "ii", "professor", "interpretation", "necessity",
    "lectures", "analysis", "effect", "preliminary", "experiments",
    "level", "note", "method", "bibliography",
    "report", "reports", "supplementary", "effects"
}

FILTER_NUMERIC_TOKENS = True

TSNE_PERPLEXITY = 30
TSNE_ITER = 1500
TSNE_LEARNING_RATE = "auto"

AUTO_SELECT_K = True
K_GRID = [8, 10, 12, 15, 18, 20, 25, 30]
TOP_N_OVERLAP = 12

SUPER_AUTO = True
SUPER_MIN = 3
DENDRO_METHOD = "average"

TARGET_JOURNALS = ["AJP", "JEP"]

TOPIC_LABELS = {
    0: "paired association",
    1: "memory",
    2: "reinforcement",
    3: "recall",
    4: "reaction time",
    5: "vision",
    6: "stimulus,",
    7: "discrimination",
    8: "conditioning",
    9: "inhibition",
    10: "learning function",
    11: "serial learning",
    12: "perception",
    13: "response",
    14: "concept",
    15: "recognition",
    16: "verbal learning",
    17: "retention",
    18: "judgement",
    19: "performance",
    20: "size, movement, color",
    21: "extinction",
    22: "transfer learning",
    23: "information processing",
    24: "conditioned response",
    25: "audition",
    26: "generalization",
    27: "probability learning",
    28: "other behavior",
    29: "test theory",
}

SUPER_LABELS = {
    0: "memory",
    1: "information processing",
    2: "retention",
    3: "test and theory",
    4: "perception",
    5: "reaction time",
    6: "learning function",
    7: "learning",
    8: "generalization",
    9: "conditioning",
    10: "reinforcement",
    11: "inhibition",
}

TOP_PAPERS_PER_DECADE = 10
LEARNING_SUPER_SET = {"learning", "conditioning", "reinforcement", "generalization", "learning function"}
COGNITIVE_SUPER_SET = {"memory", "information processing"}

DECADE_DENSITY_WINDOWS = [
    (1940, 1949, "1940-49"),
    (1950, 1959, "1950-59"),
    (1960, 1969, "1960-69"),
    (1970, 1980, "1970-80"),
]
DECADE_PALETTE = {
    "1940-49": "#0072B2",
    "1950-59": "#D55E00",
    "1960-69": "#009E73",
    "1970-80": "#CC79A7",
}

CSV_DIR = os.path.join(OUTDIR, "csv")
NOT_USED_DIR = os.path.join(OUTDIR, "not_used")

try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, leaves_list
    HAVE_SCIPY = True
except Exception as e:
    HAVE_SCIPY = False
    SCIPY_IMPORT_ERROR = e

rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

USED_FIG_DPI = 600
UNUSED_FIG_DPI = 300
EPS = 1e-12


def prepare_output_dirs():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(NOT_USED_DIR, exist_ok=True)


def save_csv(df_obj, filename, index=False):
    path = os.path.join(CSV_DIR, filename)
    df_obj.to_csv(path, index=index)
    return path


def save_used_figure(fig, filename):
    path = os.path.join(OUTDIR, filename)
    fig.savefig(path, dpi=USED_FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_unused_figure(fig, filename):
    path = os.path.join(NOT_USED_DIR, filename)
    fig.savefig(path, dpi=UNUSED_FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    return ax


def lighten(hex_color, amt):
    r, g, b = to_rgb(hex_color)
    r = r + (1 - r) * amt
    g = g + (1 - g) * amt
    b = b + (1 - b) * amt
    return to_hex((r, g, b))


def top_terms_from_row(row_vec, vocab, topn=TOP_TERMS):
    idx = np.argsort(row_vec)[::-1][:min(topn, row_vec.shape[0])]
    return ", ".join([str(vocab[i]) for i in idx])


def topic_label(topic_id):
    return TOPIC_LABELS.get(int(topic_id), f"topic {topic_id}")


def super_label_from_sid(sid):
    return SUPER_LABELS.get(int(sid), f"super {sid}")


def super_label_from_name(name):
    sid = int(str(name).replace("S", ""))
    return super_label_from_sid(sid)


# ============================================================
# 1. データ解析
# ============================================================

def is_name_lifespan_title(title):
    t = str(title).strip()
    pattern = r"^[A-Z][A-Za-z'\-.]+(?:\s+[A-Z][A-Za-z'\-.]+)+\s*:\s*(?:born\s+)?\d{4}\s*[-–—]\s*\d{4}\.?$"
    return re.fullmatch(pattern, t) is not None



def load_input():
    assert os.path.exists(INPUT_CSV), f"'{INPUT_CSV}' が見つからない。title,year 列のCSV/Excelを用意して。"
    if INPUT_CSV.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(INPUT_CSV)
    else:
        df = pd.read_csv(INPUT_CSV)

    assert {"title", "year", "journal"}.issubset(df.columns), "ファイルは 'title','year','journal' の3列が必要。"

    df = df.dropna(subset=["title", "year"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df["year"] = df["year"].astype(int)
    df["journal"] = df["journal"].astype(str)

    duplicate_mask = df.duplicated(subset=["title"], keep="first")
    removed_duplicates = df.loc[duplicate_mask].copy()
    df = df.loc[~duplicate_mask].copy()

    name_lifespan_mask = df["title"].apply(is_name_lifespan_title)
    removed_name_lifespan = df.loc[name_lifespan_mask].copy()
    df = df.loc[~name_lifespan_mask].copy()

    if len(removed_duplicates) > 0:
        save_csv(removed_duplicates, "removed_duplicate_titles.csv", index=False)
    if len(removed_name_lifespan) > 0:
        save_csv(removed_name_lifespan, "removed_name_lifespan_titles.csv", index=False)

    print(f"[PREPROCESS] removed duplicate titles: {len(removed_duplicates)}")
    print(f"[PREPROCESS] removed name+lifetime titles: {len(removed_name_lifespan)}")
    print(f"[PREPROCESS] remaining papers: {len(df)}")
    return df


def compute_tfidf(df):
    base_stop = set()
    if LANG.lower().startswith("en"):
        base_stop = set(ENGLISH_STOP_WORDS)

    custom_stop = {w.lower() for w in CUSTOM_STOPWORDS}
    stop_words_final = sorted(base_stop.union(custom_stop))

    token_pat = r"(?u)\b\w\w+\b"
    if FILTER_NUMERIC_TOKENS:
        token_pat = r"(?u)\b[^\W\d_][\w\-]*[A-Za-z]\b"

    vect = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, NGRAM_MAX),
        max_features=MAX_FEATURES,
        stop_words=stop_words_final if len(stop_words_final) > 0 else None,
        dtype=np.float32,
        token_pattern=token_pat,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X = vect.fit_transform(df["title"])
    vocab = vect.get_feature_names_out()
    return vect, X, vocab


def nmf_fit_with_compat(X, k):
    try:
        model = NMF(
            n_components=k,
            init="nndsvda",
            random_state=SEED,
            max_iter=2000,
            l1_ratio=0.25,
            alpha_W=0.0,
            alpha_H=0.0
        )
    except TypeError:
        model = NMF(
            n_components=k,
            init="nndsvda",
            random_state=SEED,
            max_iter=2000
        )
    Wk = model.fit_transform(X)
    Hk = model.components_
    return model, Wk, Hk


def jaccard_overlap_of_topics(Hk, top_n=TOP_N_OVERLAP):
    V = Hk.shape[1]
    idx_top = np.argsort(Hk, axis=1)[:, ::-1][:, :min(top_n, V)]
    sets = [set(idx_top[i]) for i in range(Hk.shape[0])]
    J = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = len(sets[i] & sets[j])
            uni = len(sets[i] | sets[j])
            if uni > 0:
                J.append(inter / uni)
    return np.mean(J) if len(J) else 0.0


def doc_dominance(Wk):
    Wn = Wk / (Wk.sum(axis=1, keepdims=True) + EPS)
    return float(np.median(Wn.max(axis=1)))


def topic_sparsity(Hk):
    P = Hk / (Hk.sum(axis=1, keepdims=True) + EPS)
    ent = -(P * np.log(P + EPS)).sum(axis=1) / np.log(P.shape[1])
    return float(1.0 - np.mean(ent))


def elbow_k(xs, ys):
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    p1 = np.array([xs[0], ys[0]])
    p2 = np.array([xs[-1], ys[-1]])
    v = p2 - p1
    v /= np.linalg.norm(v) + 1e-12
    d = np.abs((np.vstack([xs, ys]).T - p1) @ np.array([-v[1], v[0]]))
    return int(xs[np.argmax(d)])


def run_nmf_selection(X):
    metrics = []
    cache = {}

    for k in K_GRID:
        model, Wk, Hk = nmf_fit_with_compat(X, k)
        cache[k] = (model, Wk, Hk)
        rec = model.reconstruction_err_ / X.shape[0]
        jac = jaccard_overlap_of_topics(Hk, TOP_N_OVERLAP)
        dom = doc_dominance(Wk)
        spc = topic_sparsity(Hk)
        metrics.append({
            "K": k,
            "recon": rec,
            "jaccard": jac,
            "dominance": dom,
            "sparsity": spc
        })

    met = pd.DataFrame(metrics)
    save_csv(met, "nmf_k_sweep_metrics.csv", index=False)

    K_elbow = elbow_k(met["K"].values, met["recon"].values)

    cand = met.sort_values(["recon", "jaccard"]).copy()
    cand["ok"] = (cand["jaccard"] <= 0.40) & (cand["dominance"].between(0.30, 0.75))
    if AUTO_SELECT_K and cand["ok"].any():
        K = int(cand[cand["ok"]].iloc[0]["K"])
    else:
        K = K_elbow

    nmf, W, H = nmf_fit_with_compat(X, K)
    return met, K, nmf, W, H


def auto_n_super_from_linkage(Z, k, mn, mx):
    if Z.shape[0] < 2:
        return max(mn, 2)
    d = Z[:, 2]
    rel = (d[1:] - d[:-1]) / (d[:-1] + EPS)
    idx = int(np.argmax(rel)) + 1
    n = int(np.clip(k - idx, mn, mx))
    return max(n, mn)


def compute_super_topics(df, W, H, K, vocab):
    if not HAVE_SCIPY:
        raise ImportError(f"SciPy が必要。import error: {SCIPY_IMPORT_ERROR}")

    Hn = H / (H.sum(axis=1, keepdims=True) + EPS)
    SUPER_MAX = min(12, K)

    dv = pdist(Hn, metric="cosine")
    Z = linkage(dv, method=DENDRO_METHOD)
    N_SUPER = auto_n_super_from_linkage(Z, K, SUPER_MIN, SUPER_MAX) if SUPER_AUTO else min(max(SUPER_MIN, 3), SUPER_MAX)
    labels = fcluster(Z, t=N_SUPER, criterion="maxclust")

    super_ids = sorted(np.unique(labels))
    H_super = np.vstack([H[labels == sid].sum(axis=0) for sid in super_ids])
    W_super = np.hstack([W[:, labels == sid].sum(axis=1, keepdims=True) for sid in super_ids])

    super_summaries = []
    for idx_s, sid in enumerate(super_ids):
        topic_ids = np.where(labels == sid)[0].tolist()
        super_summaries.append({
            "super_id": int(sid),
            "super_label": super_label_from_sid(idx_s),
            "n_topics": int(len(topic_ids)),
            "topic_ids": ",".join(map(str, topic_ids)),
            "top_terms": top_terms_from_row(H_super[idx_s], vocab, topn=TOP_TERMS)
        })
    save_csv(pd.DataFrame(super_summaries), "super_topics_summary.csv", index=False)

    topic_rows = []
    for t in range(K):
        sid = int(labels[t])
        sid_index = super_ids.index(sid)
        topic_rows.append({
            "topic": t,
            "topic_label": topic_label(t),
            "super_id": sid,
            "super_label": super_label_from_sid(sid_index),
            "topic_top_terms": top_terms_from_row(H[t], vocab, topn=TOP_TERMS)
        })
    save_csv(pd.DataFrame(topic_rows), "topic_to_super_map.csv", index=False)

    domin_idx = W_super.argmax(axis=1)
    domin_sid = [int(super_ids[i]) for i in domin_idx]
    domin_w = W_super.max(axis=1)

    doc_out = df[["title", "year", "journal"]].copy()
    for j, sid in enumerate(super_ids):
        doc_out[super_label_from_sid(j)] = W_super[:, j]
    doc_out["dominant_super"] = [super_label_from_sid(super_ids.index(sid)) for sid in domin_sid]
    doc_out["dominant_super_weight"] = domin_w
    save_csv(doc_out, "doc_super_weights.csv", index=False)

    years_full = pd.Series(range(int(df["year"].min()), int(df["year"].max()) + 1), name="year")
    year_super_sum = pd.concat([df[["year"]].reset_index(drop=True), pd.DataFrame(W_super)], axis=1)
    year_super_sum.columns = ["year"] + [f"super_{sid}" for sid in super_ids]
    trend = year_super_sum.groupby("year").sum().reindex(years_full).fillna(0.0)
    trend_ma = trend.rolling(window=max(1, TREND_WINDOW), center=True, min_periods=1).mean()
    save_csv(trend_ma, "super_topic_trends.csv", index=True)

    return {
        "Z": Z,
        "labels": labels,
        "N_SUPER": N_SUPER,
        "super_ids": super_ids,
        "H_super": H_super,
        "W_super": W_super,
        "trend_sum": trend,
        "trend_sum_ma": trend_ma,
    }


def build_color_of_super(super_ids, H_super):
    def cb_palette(n):
        base = [
            "#1f77b4", "#d95f02", "#1b9e77", "#cc79a7", "#e69f00", "#56b4e9",
            "#e6d93c", "#000000", "#4c3b9f", "#8ecae6", "#228b3c", "#b44aa0"
        ]
        if n <= len(base):
            return base[:n]

        out = list(base)
        while len(out) < n:
            i = len(out) + 1
            h = (i * 0.61803398875) % 1.0
            if 0.12 < h < 0.18:
                h = (h + 0.22) % 1.0
            r, g, b = colorsys.hls_to_rgb(h, 0.55, 0.75)
            out.append(to_hex((r, g, b)))
        return out[:n]

    pal = cb_palette(len(super_ids))
    return {sid: pal[i] for i, sid in enumerate(super_ids)}


def compute_tsne_inputs(df, W_super, super_ids):
    doc_super_all = df[["year"]].reset_index(drop=True).copy()
    for j, sid in enumerate(super_ids):
        doc_super_all[f"super_{sid}"] = W_super[:, j]

    y_super = doc_super_all.groupby("year", as_index=False).mean(numeric_only=True)
    feature_cols = [f"super_{sid}" for sid in super_ids]
    X_y = y_super[feature_cols].values

    tsne_y = TSNE(
        n_components=2,
        perplexity=min(50, max(5, len(y_super) - 2)),
        init="random",
        learning_rate=200.0,
        random_state=SEED
    )
    Z_y = tsne_y.fit_transform(X_y)
    y_super["tsne_x"] = Z_y[:, 0]
    y_super["tsne_y"] = Z_y[:, 1]

    doc_super = df[["year", "journal"]].reset_index(drop=True).copy()
    for j, sid in enumerate(super_ids):
        doc_super[f"super_{sid}"] = W_super[:, j]

    yj_super = doc_super.groupby(["year", "journal"], as_index=False).mean(numeric_only=True)
    yj_super_filt = yj_super[yj_super["journal"].isin(TARGET_JOURNALS)].copy()
    X_yj = yj_super_filt[feature_cols].values

    tsne_yj = TSNE(
        n_components=2,
        perplexity=min(50, max(5, len(yj_super_filt) - 2)),
        init="random",
        learning_rate=200.0,
        random_state=SEED
    )
    Z_yj = tsne_yj.fit_transform(X_yj)
    yj_super_filt["tsne_x"] = Z_yj[:, 0]
    yj_super_filt["tsne_y"] = Z_yj[:, 1]

    return y_super, yj_super_filt


def compute_super_trend_analysis(df, W, W_super, H_super, super_ids, labels, vocab, H):
    N_SUPER = len(super_ids)

    WnS = W_super / (W_super.sum(axis=1, keepdims=True) + EPS)
    dts = pd.DataFrame(WnS, columns=[f"S{i}" for i in range(N_SUPER)])
    dts["year"] = df["year"].values
    year_super = dts.groupby("year").mean().sort_index()
    year_super_ma = year_super.rolling(TREND_WINDOW, min_periods=1, center=True).mean()
    years = year_super_ma.index.values.astype(int)

    Smat = year_super_ma[[f"S{i}" for i in range(N_SUPER)]].values.T
    Smat = (Smat - Smat.mean(axis=1, keepdims=True)) / (Smat.std(axis=1, keepdims=True) + 1e-12)
    Smat = Smat.astype(np.float64)

    dvec = pdist(Smat, metric="correlation").astype(np.float64)
    D = squareform(dvec)
    ZL_sc = linkage(dvec, method="average", optimal_ordering=True)

    K_MIN, K_MAX, FORCE_K = 4, min(12, N_SUPER), None
    if FORCE_K is not None:
        trend_labels_sc = fcluster(ZL_sc, FORCE_K, criterion="maxclust")
    else:
        best_labels, best_obj = None, -1
        for k in range(K_MIN, K_MAX + 1):
            labs = fcluster(ZL_sc, k, criterion="maxclust")
            try:
                sc = silhouette_score(D, labs, metric="precomputed")
            except Exception:
                sc = -1
            cnts = np.bincount(labs)[1:]
            frac_singletons = (cnts == 1).mean() if cnts.size else 0.0
            obj = sc - 0.03 * frac_singletons
            if obj > best_obj:
                best_labels = labs
                best_obj = obj
        trend_labels_sc = best_labels

    save_csv(pd.DataFrame({
        "super": [f"S{i}" for i in range(N_SUPER)],
        "super_label": [super_label_from_sid(i) for i in range(N_SUPER)],
        "trend_cluster": trend_labels_sc
    }), "super_trend_clusters.csv", index=False)

    clusters = np.sort(np.unique(trend_labels_sc))
    medoids, rep_terms, cluster_peaks = [], [], []
    for c in clusters:
        idx = np.where(trend_labels_sc == c)[0]
        subD = D[np.ix_(idx, idx)]
        rep = idx[np.argmin(subD.sum(axis=1))]
        medoids.append(int(rep))
        top_idx = np.argsort(H_super[rep])[::-1][:min(10, H_super.shape[1])]
        rep_terms.append(", ".join(vocab[top_idx]))
        mean_curve = year_super_ma.iloc[:, idx].mean(axis=1).values
        cluster_peaks.append(int(years[np.argmax(mean_curve)]))
    order_c = clusters[np.argsort(cluster_peaks)]

    rows_out = []
    BASE = year_super_ma.copy()
    for s in range(N_SUPER):
        y = BASE.iloc[:, s].values.astype(float)
        y = np.nan_to_num(y, nan=0.0)
        j = int(np.argmax(y))
        p1_year, p1_val = int(years[j]), float(y[j])
        left = max(0, j - max(3, TREND_WINDOW // 2))
        right = min(len(years) - 1, j + max(3, TREND_WINDOW // 2))
        rows_out.append({
            "super": f"S{s}",
            "super_label": super_label_from_sid(s),
            "peak_year": p1_year,
            "peak_value": p1_val,
            "peak_start_year": int(years[left]),
            "peak_end_year": int(years[right]),
            "top_terms": ", ".join(vocab[np.argsort(H_super[s])[::-1][:10]])
        })
    peak_df = pd.DataFrame(rows_out).sort_values("peak_year").reset_index(drop=True)
    q1, q2 = np.quantile(years, [1 / 3, 2 / 3])
    peak_df["era"] = peak_df["peak_year"].apply(lambda y: "early" if y <= q1 else ("mid" if y <= q2 else "late"))
    save_csv(peak_df, "super_topic_peak_summary.csv", index=False)

    Wn_topic = W / (W.sum(axis=1, keepdims=True) + EPS)
    dt_topic = pd.DataFrame(Wn_topic, columns=[f"T{i}" for i in range(W.shape[1])])
    dt_topic["year"] = df["year"].values
    year_topic = dt_topic.groupby("year").mean().sort_index()
    year_topic_ma = year_topic.rolling(TREND_WINDOW, min_periods=1, center=True).mean()

    topics_by_super = {sid: np.where(labels == sid)[0].tolist() for sid in super_ids}

    year_super_pct = year_super * 100.0
    year_super_pct_ma = year_super_pct.rolling(window=TREND_WINDOW, min_periods=1, center=True).mean()

    return {
        "year_super": year_super,
        "year_super_ma": year_super_ma,
        "year_topic": year_topic,
        "year_topic_ma": year_topic_ma,
        "years": years,
        "trend_labels_sc": trend_labels_sc,
        "clusters": clusters,
        "medoids": medoids,
        "rep_terms": rep_terms,
        "cluster_peaks": cluster_peaks,
        "order_c": order_c,
        "peak_df": peak_df,
        "topics_by_super": topics_by_super,
        "year_super_pct_ma": year_super_pct_ma,
        "BASE": BASE,
    }



def year_summary_with_ci(df_obj, value_col):
    summary = (
        df_obj.groupby("year")[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("year")
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    summary["ci"] = 1.96 * summary["se"]
    summary.loc[summary["count"] <= 1, "ci"] = 0.0
    summary["ci_low"] = summary["mean"] - summary["ci"]
    summary["ci_high"] = summary["mean"] + summary["ci"]
    return summary


def yearly_variance_summary(df_obj, value_col):
    summary = (
        df_obj.groupby("year")[value_col]
        .agg(["var", "count"])
        .reset_index()
        .sort_values("year")
    )
    summary["var"] = summary["var"].fillna(0.0)
    return summary


def make_decade_label(year):
    decade_start = int(year) // 10 * 10
    return f"{decade_start}s"


def compute_document_mixing_profiles(df, W, W_super, H, super_ids, labels):
    n_docs, K = W.shape
    if len(super_ids) < 10:
        raise ValueError("Figure 4c/4d に必要な super-topic 数が不足している。")

    Wn_topic = W / (W.sum(axis=1, keepdims=True) + EPS)
    Wn_super = W_super / (W_super.sum(axis=1, keepdims=True) + EPS)
    Hn = H / (H.sum(axis=1, keepdims=True) + EPS)
    topic_distance = squareform(pdist(Hn, metric="cosine"))
    np.fill_diagonal(topic_distance, 0.0)

    hhi = np.sum(Wn_topic ** 2, axis=1)
    mixing_distance = 0.5 * np.einsum("ij,jk,ik->i", Wn_topic, topic_distance, Wn_topic)

    topic_rank = np.argsort(Wn_topic, axis=1)[:, ::-1]
    top1_topic = topic_rank[:, 0]
    top2_topic = topic_rank[:, 1]
    top1_share = Wn_topic[np.arange(n_docs), top1_topic]
    top2_share = Wn_topic[np.arange(n_docs), top2_topic]
    top2_distance = topic_distance[top1_topic, top2_topic]

    super_rank = np.argsort(Wn_super, axis=1)[:, ::-1]
    dominant_super_idx = super_rank[:, 0]
    second_super_idx = super_rank[:, 1]
    dominant_super_share = Wn_super[np.arange(n_docs), dominant_super_idx]
    second_super_share = Wn_super[np.arange(n_docs), second_super_idx]

    dominant_super_label = [super_label_from_sid(int(i)) for i in dominant_super_idx]
    second_super_label = [super_label_from_sid(int(i)) for i in second_super_idx]

    learning_super_indices = [i for i in range(len(super_ids)) if super_label_from_sid(i) in LEARNING_SUPER_SET]
    cognitive_super_indices = [i for i in range(len(super_ids)) if super_label_from_sid(i) in COGNITIVE_SUPER_SET]

    learning_share = Wn_super[:, learning_super_indices].sum(axis=1)
    cognitive_share = Wn_super[:, cognitive_super_indices].sum(axis=1)

    doc_profile = df[["title", "year", "journal"]].copy()
    doc_profile["decade_start"] = (doc_profile["year"] // 10) * 10
    doc_profile["decade_label"] = doc_profile["year"].apply(make_decade_label)
    doc_profile["HHI"] = hhi
    doc_profile["M"] = mixing_distance
    doc_profile["dominant_topic"] = top1_topic
    doc_profile["dominant_topic_label"] = [topic_label(x) for x in top1_topic]
    doc_profile["dominant_topic_share"] = top1_share
    doc_profile["second_topic"] = top2_topic
    doc_profile["second_topic_label"] = [topic_label(x) for x in top2_topic]
    doc_profile["second_topic_share"] = top2_share
    doc_profile["top2_topic_distance"] = top2_distance
    doc_profile["dominant_super"] = dominant_super_label
    doc_profile["dominant_super_share"] = dominant_super_share
    doc_profile["second_super"] = second_super_label
    doc_profile["second_super_share"] = second_super_share
    doc_profile["learning_share"] = learning_share
    doc_profile["cognitive_share"] = cognitive_share
    doc_profile["is_learning_dominant"] = doc_profile["dominant_super"].isin(LEARNING_SUPER_SET)
    doc_profile["is_cognitive_dominant"] = doc_profile["dominant_super"].isin(COGNITIVE_SUPER_SET)

    for topic_idx in range(K):
        doc_profile[f"topic_{topic_idx}_share"] = Wn_topic[:, topic_idx]
    for super_idx in range(len(super_ids)):
        doc_profile[f"super_{super_label_from_sid(super_idx)}_share"] = Wn_super[:, super_idx]

    save_csv(doc_profile, "doc_topic_mixing_profile.csv", index=False)

    yearly_hhi = year_summary_with_ci(doc_profile, "HHI")
    yearly_hhi["metric"] = "HHI"
    yearly_hhi_var = yearly_variance_summary(doc_profile, "HHI")
    save_csv(yearly_hhi, "yearly_hhi_summary.csv", index=False)
    save_csv(yearly_hhi_var, "yearly_hhi_variance_summary.csv", index=False)

    learning_dom = doc_profile[doc_profile["is_learning_dominant"]].copy()
    yearly_learning_cognition = year_summary_with_ci(learning_dom, "cognitive_share") if len(learning_dom) else pd.DataFrame(columns=["year", "mean", "std", "count", "se", "ci", "ci_low", "ci_high"])
    save_csv(yearly_learning_cognition, "yearly_learning_dominant_cognitive_mix_summary.csv", index=False)

    cognitive_dom = doc_profile[doc_profile["is_cognitive_dominant"]].copy()
    yearly_cognition_learning = year_summary_with_ci(cognitive_dom, "learning_share") if len(cognitive_dom) else pd.DataFrame(columns=["year", "mean", "std", "count", "se", "ci", "ci_low", "ci_high"])
    save_csv(yearly_cognition_learning, "yearly_cognitive_dominant_learning_mix_summary.csv", index=False)

    typical_by_decade = (
        doc_profile.sort_values(["decade_start", "HHI", "M"], ascending=[True, False, True])
        .groupby("decade_start", group_keys=False)
        .head(TOP_PAPERS_PER_DECADE)
        .copy()
    )
    typical_by_decade["rank_within_decade"] = typical_by_decade.groupby("decade_start")["HHI"].rank(method="first", ascending=False).astype(int)
    typical_by_decade["paper_type"] = "typical"
    save_csv(typical_by_decade, "typical_papers_by_decade.csv", index=False)

    mixed_by_decade = (
        doc_profile.sort_values(["decade_start", "HHI", "M"], ascending=[True, True, False])
        .groupby("decade_start", group_keys=False)
        .head(TOP_PAPERS_PER_DECADE)
        .copy()
    )
    mixed_by_decade["rank_within_decade"] = mixed_by_decade.groupby("decade_start")["HHI"].rank(method="first", ascending=True).astype(int)
    mixed_by_decade["paper_type"] = "mixed"
    save_csv(mixed_by_decade, "mixed_papers_by_decade.csv", index=False)

    return {
        "doc_profile": doc_profile,
        "yearly_hhi": yearly_hhi,
        "yearly_hhi_var": yearly_hhi_var,
        "yearly_learning_cognition": yearly_learning_cognition,
        "yearly_cognition_learning": yearly_cognition_learning,
        "typical_by_decade": typical_by_decade,
        "mixed_by_decade": mixed_by_decade,
    }


# ============================================================
# 2. Figure 描画
# ============================================================

def plot_nmf_sweep_not_used(met, K):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))

    ax = axes[0]
    ax.plot(met["K"], met["recon"], marker="o", linewidth=1.6, markersize=4)
    ax.axvline(K, linestyle="--", color="0.5", linewidth=1.0)
    ax.set_xlabel("Number of topics (K)")
    ax.set_ylabel("Reconstruction error / N")
    style_axis(ax)

    ax = axes[1]
    ax.plot(met["K"], met["jaccard"], marker="o", linewidth=1.4, markersize=4, label="Jaccard overlap")
    ax.plot(met["K"], met["dominance"], marker="o", linewidth=1.4, markersize=4, label="Document dominance")
    ax.plot(met["K"], met["sparsity"], marker="o", linewidth=1.4, markersize=4, label="Topic sparsity")
    ax.axvline(K, linestyle="--", color="0.5", linewidth=1.0)
    ax.set_xlabel("Number of topics (K)")
    ax.set_ylabel("Metric value")
    style_axis(ax)
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    save_unused_figure(fig, "nmf_k_sweep_plots.png")


def plot_cropped_dendrogram_not_used(Z, K):
    fig, ax = plt.subplots(figsize=(max(8, K * 0.35), 5.2))
    dendrogram(
        Z,
        labels=[f"t{i}" for i in range(K)],
        leaf_rotation=90,
        leaf_font_size=8,
        distance_sort="descending",
        ax=ax
    )
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Cosine distance")
    style_axis(ax)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    save_unused_figure(fig, "topic_dendrogram_cropped.png")


def plot_figure2_dendrogram(Z, K, labels, super_ids, color_of_super):
    node_supers_cache = {}

    def leaf_super_id(leaf_id):
        return int(labels[leaf_id])

    def node_supers(node_id):
        if node_id in node_supers_cache:
            return node_supers_cache[node_id]
        if node_id < K:
            s = {leaf_super_id(node_id)}
        else:
            l = int(Z[node_id - K, 0])
            r = int(Z[node_id - K, 1])
            s = node_supers(l) | node_supers(r)
        node_supers_cache[node_id] = s
        return s

    def link_color_func(node_id):
        s = node_supers(node_id)
        if len(s) == 1:
            sid = next(iter(s))
            return color_of_super.get(int(sid), "#999999")
        return "#999999"

    YMIN = 0.60
    YMAX = 0.999

    fig, ax = plt.subplots(figsize=(max(8, K * 0.35), 5.5))
    dobj = dendrogram(
        Z,
        labels=[topic_label(i) for i in range(K)],
        color_threshold=0.0,
        link_color_func=link_color_func,
        leaf_rotation=90,
        leaf_font_size=8,
        distance_sort="descending",
        ax=ax
    )

    xpos_to_leaf = {5.0 + 10.0 * i: leaf_id for i, leaf_id in enumerate(dobj["leaves"])}
    for xs, ys in zip(dobj["icoord"], dobj["dcoord"]):
        if ys[0] == 0.0:
            x, leaf = xs[0], xpos_to_leaf.get(xs[0])
            if leaf is not None:
                ax.plot([x, x], [max(YMIN, ys[0]), ys[1]], lw=2.2, color=color_of_super[labels[leaf]], zorder=3)
        if ys[3] == 0.0:
            x, leaf = xs[3], xpos_to_leaf.get(xs[3])
            if leaf is not None:
                ax.plot([x, x], [max(YMIN, ys[3]), ys[2]], lw=2.2, color=color_of_super[labels[leaf]], zorder=3)

    ax.set_yscale("logit")
    ax.set_ylim(YMIN, YMAX)
    ax.set_ylabel("Cosine distance")
    style_axis(ax)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    save_used_figure(fig, "figure_2_topic_dendrogram_supercolor_logit.png")


def plot_unused_super_topic_trend(trend_sum_ma, super_ids, color_of_super):
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    for j, sid in enumerate(super_ids):
        ax.plot(
            trend_sum_ma.index,
            trend_sum_ma[f"super_{sid}"],
            label=super_label_from_sid(j),
            linewidth=1.6,
            color=color_of_super[sid]
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("Sum of weights (smoothed)")
    style_axis(ax)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    save_unused_figure(fig, "super_topic_trend.png")


def plot_figure1a(y_super):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))

    ax.scatter(
        y_super["tsne_x"],
        y_super["tsne_y"],
        c=y_super["year"],
        cmap="viridis",
        s=70,
        linewidths=0.4,
        edgecolors="white"
    )

    for _, row in y_super.iterrows():
        ax.text(
            row["tsne_x"],
            row["tsne_y"],
            str(int(row["year"])),
            fontsize=7,
            ha="center",
            va="center"
        )

    y_sorted = y_super.sort_values("year")
    ax.plot(
        y_sorted["tsne_x"],
        y_sorted["tsne_y"],
        color="0.55",
        alpha=0.7,
        linewidth=1.0,
        zorder=0
    )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    style_axis(ax)
    fig.tight_layout()
    save_used_figure(fig, "figure_1a_super_tsne_yearly_alljournals.png")


def plot_figure1b(yj_super_filt):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2), sharex=True, sharey=True)

    for ax_idx, journal_name in enumerate(TARGET_JOURNALS):
        ax = axes[ax_idx]
        sub = yj_super_filt[yj_super_filt["journal"] == journal_name].copy()

        ax.scatter(
            sub["tsne_x"],
            sub["tsne_y"],
            c=sub["year"],
            cmap="viridis",
            s=70,
            linewidths=0.4,
            edgecolors="white"
        )

        for _, row in sub.iterrows():
            ax.text(
                row["tsne_x"],
                row["tsne_y"],
                str(int(row["year"])),
                fontsize=7,
                ha="center",
                va="center"
            )

        sub_sorted = sub.sort_values("year")
        ax.plot(
            sub_sorted["tsne_x"],
            sub_sorted["tsne_y"],
            color="0.55",
            alpha=0.7,
            linewidth=1.0,
            zorder=0
        )

        ax.set_xlabel("t-SNE 1")
        if ax_idx == 0:
            ax.set_ylabel("t-SNE 2")
        ax.set_title(journal_name)
        style_axis(ax)

    fig.tight_layout()
    save_used_figure(fig, "figure_1b_super_tsne_by_journal_split.png")


def plot_unused_super_each_panel_timeseries(year_super_ma, super_ids, color_of_super):
    n_supers = year_super_ma.shape[1]
    cols, rows = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.0 * rows), sharex=True)
    axes = np.array(axes).reshape(rows, cols)

    years = year_super_ma.index.values
    for i in range(n_supers):
        ax = axes.flat[i]
        y = year_super_ma.iloc[:, i].values
        ymin, ymax = 0, np.nanmax(y) + np.std(y) * 0.5
        sid = super_ids[i]
        ax.plot(years, y, color=color_of_super[sid], alpha=0.9, linewidth=1.6)
        ax.set_ylim(ymin, ymax)
        ax.set_title(super_label_from_sid(i), fontsize=9)
        style_axis(ax)
        ax.grid(alpha=0.25)

    for j in range(i + 1, rows * cols):
        axes.flat[j].axis("off")

    fig.tight_layout()
    save_unused_figure(fig, "super_each_panel_timeseries.png")


def distribute_label_positions(y_values, ymin, ymax, min_gap):
    if len(y_values) == 0:
        return np.array([])
    order = np.argsort(y_values)
    ys = np.array(y_values, dtype=float)[order]

    lower = ymin + min_gap
    upper = ymax - min_gap
    ys = np.clip(ys, lower, upper)

    for i in range(1, len(ys)):
        if ys[i] < ys[i - 1] + min_gap:
            ys[i] = ys[i - 1] + min_gap

    overflow = ys[-1] - upper
    if overflow > 0:
        ys -= overflow

    for i in range(len(ys) - 2, -1, -1):
        if ys[i] > ys[i + 1] - min_gap:
            ys[i] = ys[i + 1] - min_gap

    ys = np.clip(ys, lower, upper)
    result = np.empty_like(ys)
    result[order] = ys
    return result


def plot_figure3a(year_topic_ma, years, topics_by_super, super_ids, color_of_super):
    n_supers = len(super_ids)
    cols, rows = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.5 * rows), sharex=True)
    axes = np.array(axes).reshape(rows, cols)

    x_span = years[-1] - years[0]
    x_text = years[-1] + x_span * 0.05
    x_lim_right = years[-1] + x_span * 0.24

    for panel_idx, (ax, sid) in enumerate(zip(axes.flat, super_ids)):
        base = color_of_super[sid]
        tidx = topics_by_super[sid]

        if len(tidx) == 1:
            shades = [base]
        else:
            shades = [lighten(base, a) for a in np.linspace(0.0, 0.65, len(tidx))]

        yvals = []
        for ti in tidx:
            yvals.append(year_topic_ma[f"T{ti}"].values)
        yvals = np.vstack(yvals) if len(yvals) else np.zeros((1, len(years)))
        ymin = 0
        ymax = float(np.nanmax(yvals) + np.std(yvals) * 0.5)
        if not np.isfinite(ymax) or ymax <= ymin:
            ymax = 1.0

        end_vals = []
        line_info = []
        for j, ti in enumerate(tidx):
            y = year_topic_ma[f"T{ti}"].values
            ax.plot(years, y, color=shades[j], lw=1.6, alpha=0.95)
            end_idx = np.where(np.isfinite(y))[0][-1]
            end_vals.append(float(y[end_idx]))
            line_info.append((ti, shades[j], end_idx, float(y[end_idx])))

        min_gap = max((ymax - ymin) * 0.08, 0.004)
        label_ys = distribute_label_positions(end_vals, ymin, ymax, min_gap)

        for (ti, col, end_idx, y_end), y_lab in zip(line_info, label_ys):
            ax.plot([years[end_idx], x_text - x_span * 0.01], [y_end, y_lab], color=col, lw=0.8, alpha=0.8)
            ax.text(
                x_text,
                y_lab,
                topic_label(ti),
                fontsize=7,
                color=col,
                ha="left",
                va="center",
                clip_on=False
            )

        ax.set_title(f"{super_label_from_sid(panel_idx)}  (topics={len(tidx)})", fontsize=10, color=base)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(years[0], x_lim_right)
        style_axis(ax)
        ax.grid(alpha=0.25)

    for k in range(len(super_ids), rows * cols):
        axes.flat[k].axis("off")

    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    save_used_figure(fig, "figure_3_panels_topic_timeseries_per_super.png")


def plot_figure3b(BASE, peak_df, years):
    order_rows = peak_df["super"].tolist()
    HM = BASE[order_rows].T.values
    HM = (HM - HM.mean(axis=1, keepdims=True)) / (HM.std(axis=1, keepdims=True) + 1e-12)

    y_labels = [super_label_from_name(x) for x in order_rows]

    fig, ax = plt.subplots(figsize=(11, max(5, HM.shape[0] * 0.35 + 2)))
    im = ax.imshow(HM, aspect="auto", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("z score per super-topic")
    ax.set_yticks(np.arange(HM.shape[0]))
    ax.set_yticklabels(y_labels, fontsize=8)
    xt = np.linspace(0, len(years) - 1, num=min(12, len(years))).astype(int)
    ax.set_xticks(xt)
    ax.set_xticklabels(years[xt], rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Super-topic")
    fig.tight_layout()
    save_used_figure(fig, "figure_4a_super_year_heatmap_sorted_by_peak.png")


def plot_figure3c(year_super_pct_ma, years, super_ids, color_of_super):
    cols_order = [f"S{i}" for i in range(len(super_ids))]
    Y = [year_super_pct_ma[c].values for c in cols_order]
    colors = [color_of_super[sid] for sid in super_ids]
    labels = [super_label_from_sid(i) for i in range(len(super_ids))]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    polys = ax.stackplot(
        years,
        Y,
        colors=colors,
        alpha=0.98
    )

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_ylim(0, 100)
    ax.set_xlim(years[0], years[-1])
    ax.margins(x=0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Super-topic share per paper (%)")
    style_axis(ax)

    fig.subplots_adjust(bottom=0.26)
    fig.legend(
        polys,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        fontsize=8
    )
    save_used_figure(fig, "figure_4b_super_percent_stacked.png")



def plot_yearly_scatter_mean_ci(ax, df_obj, value_col, y_label, line_color="#d62728"):
    ax.scatter(
        df_obj["year"],
        df_obj[value_col],
        s=10,
        color="#bfbfbf",
        alpha=0.35,
        linewidths=0,
        zorder=1
    )
    summary = year_summary_with_ci(df_obj, value_col)
    ax.fill_between(
        summary["year"],
        summary["ci_low"],
        summary["ci_high"],
        color=line_color,
        alpha=0.18,
        linewidth=0,
        zorder=2
    )
    ax.plot(
        summary["year"],
        summary["mean"],
        color=line_color,
        linewidth=2.0,
        zorder=3
    )
    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    style_axis(ax)
    return summary


def plot_figure4a(doc_profile):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    plot_yearly_scatter_mean_ci(ax, doc_profile, "HHI", "Herfindahl index")
    fig.tight_layout()
    save_used_figure(fig, "figure_5a_hhi_timeseries.png")


def plot_not_used_hhi_variance(doc_profile):
    # 参考用: 各年のHHI分散（本文では使わない想定）
    summary = yearly_variance_summary(doc_profile, "HHI")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(summary["year"], summary["var"], color="#d62728", linewidth=2.0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Variance of Herfindahl index")
    style_axis(ax)
    fig.tight_layout()
    save_unused_figure(fig, "hhi_variance_timeseries.png")


def plot_figure4c(doc_profile):
    sub = doc_profile[doc_profile["is_learning_dominant"]].copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    plot_yearly_scatter_mean_ci(
        ax,
        sub,
        "cognitive_share",
        "Memory + information processing share"
    )
    fig.tight_layout()
    save_used_figure(fig, "figure_5b_learning_dominant_papers_cognitive_mix.png")


def plot_figure4d(doc_profile):
    sub = doc_profile[doc_profile["is_cognitive_dominant"]].copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    plot_yearly_scatter_mean_ci(
        ax,
        sub,
        "learning_share",
        "Learning-related share"
    )
    fig.tight_layout()
    save_used_figure(fig, "figure_5c_cognitive_dominant_papers_learning_mix.png")


def plot_decade_density(ax, values, label, color):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= 0.0) & (vals <= 1.0)]
    if len(vals) == 0:
        return False

    x_grid = np.linspace(0.0, 1.0, 512)
    if len(vals) == 1 or np.allclose(vals, vals[0]):
        mu = float(vals[0])
        sigma = 0.03
        y = np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    else:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals)
        y = kde(x_grid)

    ax.plot(x_grid, y, color=color, linewidth=2.0, label=label)
    return True


def plot_figure4e(doc_profile):
    sub = doc_profile[doc_profile["is_learning_dominant"]].copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    has_any = False
    for start_year, end_year, label in DECADE_DENSITY_WINDOWS:
        vals = sub.loc[sub["year"].between(start_year, end_year), "cognitive_share"].values
        has_any = plot_decade_density(ax, vals, label, DECADE_PALETTE[label]) or has_any

    ax.set_xlabel("Memory + information processing share")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    style_axis(ax)
    if has_any:
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    save_used_figure(fig, "figure_5d_learning_dominant_papers_cognitive_mix_density.png")


def plot_figure4f(doc_profile):
    sub = doc_profile[doc_profile["is_cognitive_dominant"]].copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    has_any = False
    for start_year, end_year, label in DECADE_DENSITY_WINDOWS:
        vals = sub.loc[sub["year"].between(start_year, end_year), "learning_share"].values
        has_any = plot_decade_density(ax, vals, label, DECADE_PALETTE[label]) or has_any

    ax.set_xlabel("Learning-related share")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    style_axis(ax)
    if has_any:
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    save_used_figure(fig, "figure_5e_cognitive_dominant_papers_learning_mix_density.png")


def remove_obsolete_outputs():
    # 以前の命名規則で保存された画像が残っていると混乱しやすいので掃除する
    obsolete_files = [
        # Old Figure 3 naming
        os.path.join(OUTDIR, "figure_3a_panels_topic_timeseries_per_super.png"),
        os.path.join(OUTDIR, "figure_3b_super_year_heatmap_sorted_by_peak.png"),
        os.path.join(OUTDIR, "figure_3c_super_percent_stacked.png"),

        # Old Figure 4 naming (now Figure 5)
        os.path.join(OUTDIR, "figure_4a_hhi_timeseries.png"),
        os.path.join(OUTDIR, "figure_4b_learning_dominant_papers_cognitive_mix.png"),
        os.path.join(OUTDIR, "figure_4c_cognitive_dominant_papers_learning_mix.png"),
        os.path.join(OUTDIR, "figure_4d_learning_dominant_papers_cognitive_mix_density.png"),
        os.path.join(OUTDIR, "figure_4e_cognitive_dominant_papers_learning_mix_density.png"),

        # Other legacy files from earlier revisions
        os.path.join(OUTDIR, "figure_4b_hhi_variance_timeseries.png"),
        os.path.join(OUTDIR, "figure_4f_hhi_variance_timeseries.png"),
        os.path.join(OUTDIR, "figure_4f_hhi_variance_timeseries.png"),
    ]
    for path in obsolete_files:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

def main():

    prepare_output_dirs()
    remove_obsolete_outputs()

    # ===== データ解析 =====
    df = load_input()
    vect, X, vocab = compute_tfidf(df)

    met, K, nmf, W, H = run_nmf_selection(X)
    super_result = compute_super_topics(df, W, H, K, vocab)
    color_of_super = build_color_of_super(
        super_result["super_ids"],
        super_result["H_super"]
    )

    y_super, yj_super_filt = compute_tsne_inputs(
        df,
        super_result["W_super"],
        super_result["super_ids"]
    )

    trend_result = compute_super_trend_analysis(
        df,
        W,
        super_result["W_super"],
        super_result["H_super"],
        super_result["super_ids"],
        super_result["labels"],
        vocab,
        H
    )

    mixing_result = compute_document_mixing_profiles(
        df,
        W,
        super_result["W_super"],
        H,
        super_result["super_ids"],
        super_result["labels"]
    )

    # ===== Figure 描画 =====
    plot_nmf_sweep_not_used(met, K)
    plot_cropped_dendrogram_not_used(super_result["Z"], K)
    plot_unused_super_topic_trend(super_result["trend_sum_ma"], super_result["super_ids"], color_of_super)
    plot_unused_super_each_panel_timeseries(trend_result["year_super_ma"], super_result["super_ids"], color_of_super)

    plot_figure1a(y_super)
    plot_figure1b(yj_super_filt)
    plot_figure2_dendrogram(
        super_result["Z"],
        K,
        super_result["labels"],
        super_result["super_ids"],
        color_of_super
    )
    plot_figure3a(
        trend_result["year_topic_ma"],
        trend_result["years"],
        trend_result["topics_by_super"],
        super_result["super_ids"],
        color_of_super
    )
    plot_figure3b(
        trend_result["BASE"],
        trend_result["peak_df"],
        trend_result["years"]
    )
    plot_figure3c(
        trend_result["year_super_pct_ma"],
        trend_result["years"],
        super_result["super_ids"],
        color_of_super
    )
    plot_figure4a(mixing_result["doc_profile"])
    plot_figure4c(mixing_result["doc_profile"])
    plot_figure4d(mixing_result["doc_profile"])
    plot_figure4e(mixing_result["doc_profile"])
    plot_figure4f(mixing_result["doc_profile"])

    print(f"[DONE] selected K={K}")
    print(f"[DONE] figures saved in: {os.path.abspath(OUTDIR)}")
    print(f"[DONE] unused figures saved in: {os.path.abspath(NOT_USED_DIR)}")
    print(f"[DONE] csv files saved in: {os.path.abspath(CSV_DIR)}")


if __name__ == "__main__":
    main()
