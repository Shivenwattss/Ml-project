"""
Algorithm Recommender
---------------------
Supervised model that looks at an UNLABELLED 2D dataset and predicts whether
K-Means or DBSCAN will cluster it better.

Pipeline
  1. generate_dataset()   -> synthetic datasets of many shapes, with ground truth
  2. label_dataset()      -> run both algorithms (tuned using the ground truth)
                             and record which one wins on Adjusted Rand Index
  3. extract_features()   -> label-free statistics describing the dataset
  4. train_model()        -> Random Forest (+ Logistic Regression and majority baselines)
  5. run_recommender()    -> Streamlit page: performance report + live demo
"""
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    adjusted_rand_score, silhouette_score,
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "recommender_model.joblib")
DATA_PATH = os.path.join(HERE, "recommender_training_data.csv")

KINDS = ["blobs", "varied_blobs", "aniso_blobs", "moons", "circles"]
FEATURES = [
    "knn_mean", "knn_cv", "knn_tail_ratio", "frac_sparse", "pca_ratio",
    "km_best_k", "km_best_sil", "db_n_clusters", "db_noise_frac", "db_sil",
]
CLASSES = ["K-Means", "DBSCAN"]
TIE_MARGIN = 0.05          # |ARI_km - ARI_db| below this = tie (dropped from training)
PALETTE = ["#E24B4A", "#185FA5", "#1D9E75", "#EF9F27", "#9B59B6", "#E67E22",
           "#1ABC9C", "#E91E63", "#F1C40F"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET GENERATOR  (ground-truth labels, -1 = injected noise)
# ══════════════════════════════════════════════════════════════════════════════
def generate_dataset(kind, seed):
    rng = np.random.RandomState(seed)
    n = int(rng.randint(250, 551))
    if kind == "blobs":
        X, y = make_blobs(n, centers=int(rng.randint(2, 6)),
                          cluster_std=rng.uniform(0.7, 2.2), random_state=seed)
    elif kind == "varied_blobs":
        k = int(rng.randint(2, 6))
        X, y = make_blobs(n, centers=k, cluster_std=rng.uniform(0.5, 2.4, k),
                          random_state=seed)
    elif kind == "aniso_blobs":
        X, y = make_blobs(n, centers=int(rng.randint(2, 5)), cluster_std=0.8,
                          random_state=seed)
        X = X @ np.array([[0.6, -0.6], [-0.4, 0.8]])
    elif kind == "moons":
        X, y = make_moons(n, noise=rng.uniform(0.04, 0.14), random_state=seed)
    else:
        X, y = make_circles(n, noise=rng.uniform(0.03, 0.08),
                            factor=rng.uniform(0.3, 0.6), random_state=seed)
    max_noise = 0.03 if "blobs" in kind else 0.10
    n_noise = int(n * rng.uniform(0.0, max_noise))
    if n_noise:
        lo, hi = X.min(0) - 0.5, X.max(0) + 0.5
        X = np.vstack([X, rng.uniform(lo, hi, (n_noise, 2))])
        y = np.concatenate([y, np.full(n_noise, -1)])
    return StandardScaler().fit_transform(X), y


# ══════════════════════════════════════════════════════════════════════════════
# 2. LABELLING  (which algorithm wins, each tuned using ground truth)
# ══════════════════════════════════════════════════════════════════════════════
def label_dataset(X, y):
    k_true = len(set(y) - {-1})
    km = KMeans(max(k_true, 2), n_init=5, random_state=0).fit_predict(X)
    ari_km = adjusted_rand_score(y, km)
    ari_db = max(
        adjusted_rand_score(y, DBSCAN(eps=e, min_samples=ms).fit_predict(X))
        for e in np.arange(0.05, 0.61, 0.05) for ms in (4, 6)
    )
    return ari_km, ari_db


def winner_of(ari_km, ari_db):
    if ari_db - ari_km > TIE_MARGIN:
        return "DBSCAN"
    if ari_km - ari_db > TIE_MARGIN:
        return "K-Means"
    return "Tie"


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURES  (never look at ground truth)
# ══════════════════════════════════════════════════════════════════════════════
def heuristic_dbscan(X):
    """DBSCAN with eps from the 90th percentile of 5-NN distances."""
    d = NearestNeighbors(n_neighbors=6).fit(X).kneighbors(X)[0][:, 5]
    eps = float(np.percentile(d, 90))
    return DBSCAN(eps=eps, min_samples=5).fit_predict(X), eps


def best_kmeans(X):
    best_s, best_k, best_lab = -1, 2, None
    for k in range(2, 9):
        lab = KMeans(k, n_init=3, random_state=0).fit_predict(X)
        s = silhouette_score(X, lab)
        if s > best_s:
            best_s, best_k, best_lab = s, k, lab
    return best_k, best_s, best_lab


def extract_features(X):
    d5 = NearestNeighbors(n_neighbors=6).fit(X).kneighbors(X)[0][:, 5]
    med = np.median(d5)
    best_k, best_s, _ = best_kmeans(X)
    lab, _ = heuristic_dbscan(X)
    mask = lab != -1
    n_cl = len(set(lab[mask]))
    db_sil = silhouette_score(X[mask], lab[mask]) if n_cl >= 2 and mask.sum() > n_cl else 0.0
    return {
        "knn_mean": d5.mean(),
        "knn_cv": d5.std() / d5.mean(),
        "knn_tail_ratio": np.percentile(d5, 95) / med,
        "frac_sparse": float(np.mean(d5 > 2 * med)),
        "pca_ratio": PCA(2).fit(X).explained_variance_ratio_[0],
        "km_best_k": best_k,
        "km_best_sil": best_s,
        "db_n_clusters": n_cl,
        "db_noise_frac": float(np.mean(lab == -1)),
        "db_sil": db_sil,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD TRAINING DATA + TRAIN
# ══════════════════════════════════════════════════════════════════════════════
def build_training_data(n_datasets=800, progress=None):
    rows = []
    for i in range(n_datasets):
        kind = KINDS[i % len(KINDS)]
        X, y = generate_dataset(kind, seed=i)
        ari_km, ari_db = label_dataset(X, y)
        row = extract_features(X)
        row.update(kind=kind, ari_km=ari_km, ari_db=ari_db, winner=winner_of(ari_km, ari_db))
        rows.append(row)
        if progress:
            progress((i + 1) / n_datasets)
    return pd.DataFrame(rows)


def train_model(df):
    data = df[df["winner"] != "Tie"].reset_index(drop=True)
    X, y = data[FEATURES], data["winner"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
    }
    results = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        p, r, f, _ = precision_recall_fscore_support(
            yte, pred, average="macro", zero_division=0)
        results[name] = {"Accuracy": accuracy_score(yte, pred), "Precision (macro)": p,
                         "Recall (macro)": r, "F1 (macro)": f}

    rf = models["Random Forest"]
    cv = cross_val_score(
        RandomForestClassifier(n_estimators=300, random_state=42), X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42))
    pred = rf.predict(Xte)
    per_class = precision_recall_fscore_support(yte, pred, labels=CLASSES, zero_division=0)
    return {
        "model": rf,
        "results": results,
        "cm": confusion_matrix(yte, pred, labels=CLASSES),
        "per_class": pd.DataFrame(
            {"Precision": per_class[0], "Recall": per_class[1],
             "F1": per_class[2], "Support": per_class[3]}, index=CLASSES),
        "importances": pd.Series(rf.feature_importances_, index=FEATURES).sort_values(),
        "cv_mean": cv.mean(), "cv_std": cv.std(),
        "n_total": len(df), "n_ties": int((df["winner"] == "Tie").sum()),
        "n_train": len(Xtr), "n_test": len(Xte),
        "class_counts": y.value_counts().to_dict(),
        "win_by_kind": pd.crosstab(df["kind"], df["winner"]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. STREAMLIT PAGE
# ══════════════════════════════════════════════════════════════════════════════
def scatter(X, labels, title):
    fig = go.Figure()
    for l in sorted(set(labels)):
        m = labels == l
        fig.add_trace(go.Scatter(
            x=X[m, 0], y=X[m, 1], mode="markers",
            name="Noise" if l == -1 else f"Cluster {l}",
            marker=dict(size=6, color="#888" if l == -1 else PALETTE[l % len(PALETTE)],
                        symbol="x" if l == -1 else "circle")))
    fig.update_layout(template="plotly_dark", height=340, title=title,
                      margin=dict(t=50, b=20, l=20, r=20), legend=dict(orientation="h", y=-0.15))
    return fig


@st.cache_resource(show_spinner=False)
def load_or_train():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass                      # e.g. scikit-learn version mismatch -> retrain
    if os.path.exists(DATA_PATH):     # fast path: training data already generated
        df = pd.read_csv(DATA_PATH)
    else:
        bar = st.progress(0.0, text="Generating datasets and labelling them (one-off, ~2–4 min)…")
        df = build_training_data(800, progress=lambda f: bar.progress(f))
        df.to_csv(DATA_PATH, index=False)
        bar.empty()
    bundle = train_model(df)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def performance_tab(b):
    st.markdown(
        f"**Training data:** {b['n_total']} synthetic datasets → {b['n_ties']} ties removed → "
        f"{b['n_train']} train / {b['n_test']} test. "
        f"Class balance: {b['class_counts']}.")
    res = pd.DataFrame(b["results"]).T.round(3)
    st.subheader("Test-set metrics")
    st.dataframe(res, use_container_width=True)
    st.caption(f"Random Forest 5-fold cross-validated accuracy: "
               f"{b['cv_mean']:.3f} ± {b['cv_std']:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Heatmap(z=b["cm"], x=CLASSES, y=CLASSES, text=b["cm"],
                                   texttemplate="%{text}", colorscale="Blues", showscale=False))
        fig.update_layout(template="plotly_dark", height=320, title="Confusion matrix (Random Forest)",
                          xaxis_title="Predicted", yaxis_title="Actual",
                          margin=dict(t=50, b=40, l=40, r=20))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(b["per_class"].round(3), use_container_width=True)
    with c2:
        imp = b["importances"]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#1D9E75"))
        fig.update_layout(template="plotly_dark", height=320, title="Feature importance",
                          margin=dict(t=50, b=40, l=40, r=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Which algorithm actually wins, by dataset type")
    st.dataframe(b["win_by_kind"], use_container_width=True)

    st.subheader("Features used")
    st.markdown(
        "- `knn_mean`, `knn_cv`, `knn_tail_ratio`, `frac_sparse`: spread of 5-nearest-neighbour "
        "distances (density uniformity, outliers)\n"
        "- `pca_ratio`: how elongated the data is\n"
        "- `km_best_k`, `km_best_sil`: best k in 2–8 by silhouette, and that silhouette\n"
        "- `db_n_clusters`, `db_noise_frac`, `db_sil`: DBSCAN run with an automatic eps")


def demo_tab(b):
    src = st.radio("Dataset source", ["Generate a new test dataset", "Upload a CSV"], horizontal=True)
    truth = None
    if src.startswith("Generate"):
        c1, c2 = st.columns(2)
        kind = c1.selectbox("Shape", KINDS)
        seed = c2.number_input("Seed (unseen by the model)", 100000, 999999, 123456)
        X, truth = generate_dataset(kind, int(seed))
    else:
        up = st.file_uploader("CSV with numeric columns (2D plot; more than 2 columns are reduced with PCA)",
                              type="csv")
        if up is None:
            st.info("Upload a CSV to get a recommendation.")
            return
        num = pd.read_csv(up).select_dtypes("number").dropna()
        if num.shape[1] < 2 or len(num) < 30:
            st.error("Need at least 2 numeric columns and 30 rows.")
            return
        if len(num) > 1500:
            num = num.sample(1500, random_state=0)
        Z = StandardScaler().fit_transform(num.values)
        X = StandardScaler().fit_transform(PCA(2).fit_transform(Z) if Z.shape[1] > 2 else Z)

    feats = extract_features(X)
    proba = b["model"].predict_proba(pd.DataFrame([feats])[FEATURES])[0]
    classes = list(b["model"].classes_)
    pick = classes[int(np.argmax(proba))]
    st.success(f"### Recommended: **{pick}**  ({max(proba):.0%} confidence)")
    st.caption("  ·  ".join(f"{c}: {p:.0%}" for c, p in zip(classes, proba)))

    k, sil_k, km_lab = best_kmeans(X)
    db_lab, eps = heuristic_dbscan(X)
    mask = db_lab != -1
    ncl = len(set(db_lab[mask]))
    sil_d = silhouette_score(X[mask], db_lab[mask]) if ncl >= 2 else float("nan")

    c1, c2 = st.columns(2)
    c1.plotly_chart(scatter(X, km_lab, f"K-Means (k={k}, silhouette {sil_k:.2f})"),
                    use_container_width=True)
    c2.plotly_chart(scatter(X, db_lab, f"DBSCAN (eps={eps:.2f}, {ncl} clusters, silhouette {sil_d:.2f})"),
                    use_container_width=True)

    if truth is not None:
        ari_km, ari_db = label_dataset(X, truth)
        w = winner_of(ari_km, ari_db)
        verdict = "✅ correct" if w == pick else ("➖ tie, no clear winner" if w == "Tie" else "❌ wrong")
        st.markdown(f"**Ground truth check:** best-tuned ARI → K-Means {ari_km:.2f}, "
                    f"DBSCAN {ari_db:.2f}. Actual winner: **{w}**. Recommendation {verdict}.")
    with st.expander("Feature values for this dataset"):
        st.json({k_: round(float(v), 4) for k_, v in feats.items()})


def run_recommender():
    st.title("Algorithm Recommender — K-Means or DBSCAN?")
    st.caption("A Random Forest trained on hundreds of synthetic datasets predicts which "
               "clustering algorithm will work better on a new dataset.")
    if st.sidebar.button("♻ Retrain model from scratch"):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        st.cache_resource.clear()
        st.rerun()
    bundle = load_or_train()
    t1, t2 = st.tabs(["📊 Model performance", "🔮 Try it"])
    with t1:
        performance_tab(bundle)
    with t2:
        demo_tab(bundle)
