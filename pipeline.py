"""K-Means log grouping plus a supervised noisy-alert classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
LOG_TEMPLATES = [
    "api gateway returned 502 for request", "upstream service returned http 503", "api error rate exceeded threshold", "http 500 response from checkout endpoint", "gateway request failed with server error",
    "database connection pool exhausted", "postgres query duration exceeded limit", "database replica lag is high", "connection timeout while reading database", "sql transaction rolled back after lock wait",
    "worker memory usage above limit", "container reached memory pressure threshold", "process rss grew beyond budget", "node memory utilization is critical", "out of memory event on worker pod",
    "authentication token validation failed", "login failures increased for tenant", "permission denied for api client", "identity provider returned unauthorized", "invalid credential attempts detected",
    "job retry count exceeded threshold", "queue consumer processing is delayed", "background task has not completed", "scheduled job missed its deadline", "message delivery retried repeatedly",
]
FEATURES = ["repeat_count", "occurrences_per_hour", "distinct_hosts", "auto_resolved_rate", "incident_linked", "acknowledgment_rate"]


def _logs():
    services = ["api", "database", "worker", "identity", "queue"]
    return [f"service {services[j % 5]}: {template}"
            for _ in range(25) for j, template in enumerate(LOG_TEMPLATES)]


def _noisy_data(seed=SEED, rows=1200):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "repeat_count": rng.poisson(4, rows),
        "occurrences_per_hour": rng.poisson(3, rows),
        "distinct_hosts": rng.integers(1, 16, rows),
        "auto_resolved_rate": rng.beta(3, 2, rows),
        "incident_linked": rng.binomial(1, 0.32, rows),
        "acknowledgment_rate": rng.beta(2, 3, rows),
    })
    score = (0.16 * X.repeat_count + 0.26 * X.occurrences_per_hour - 0.06 * X.distinct_hosts
             + 0.8 * X.auto_resolved_rate + 0.85 * (1 - X.incident_linked)
             - 1.0 * X.acknowledgment_rate + rng.normal(0, 0.45, rows))
    return X, (score > 1.45).astype(int)


def _clusterer(seed=SEED):
    return {"vectorizer": TfidfVectorizer(ngram_range=(1, 2), min_df=1),
            "kmeans": KMeans(n_clusters=5, n_init=10, random_state=seed)}


def _noise_model(seed=SEED):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))


def sample_input():
    return {"logs": ["database connection pool exhausted", "api gateway returned 502"],
            "alert_features": {"repeat_count": 14, "occurrences_per_hour": 12, "distinct_hosts": 1,
                               "auto_resolved_rate": 0.9, "incident_linked": 0, "acknowledgment_rate": 0.1}}


def train_model(seed=SEED):
    cluster = _clusterer(seed)
    matrix = cluster["vectorizer"].fit_transform(_logs())
    cluster["kmeans"].fit(matrix)
    X, y = _noisy_data(seed)
    classifier = _noise_model(seed).fit(X, y)
    return {"cluster": cluster, "noise": classifier}


def predict(model, payload):
    texts = payload["logs"] if isinstance(payload["logs"], list) else [payload["logs"]]
    cluster = model["cluster"]
    matrix = cluster["vectorizer"].transform(texts)
    row = pd.DataFrame([payload["alert_features"]], columns=FEATURES)
    classifier = model["noise"]
    return {"clusters": [int(value) for value in cluster["kmeans"].predict(matrix)],
            "potentially_noisy": bool(classifier.predict(row)[0]),
            "noisy_probability": round(float(classifier.predict_proba(row)[0, 1]), 4)}


def evaluate(seed=SEED):
    texts = _logs()
    groups = [f"template-{j}" for _ in range(25) for j, _ in enumerate(LOG_TEMPLATES)]
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(texts, groups=groups))
    logs_train = [texts[i] for i in train_idx]
    logs_test = [texts[i] for i in test_idx]
    cluster = _clusterer(seed)
    cluster["kmeans"].fit(cluster["vectorizer"].fit_transform(logs_train))
    test_matrix = cluster["vectorizer"].transform(logs_test)
    cluster_labels = cluster["kmeans"].predict(test_matrix)
    silhouette = silhouette_score(test_matrix, cluster_labels, metric="cosine")
    X, y = _noisy_data(seed)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    classifier = _noise_model(seed).fit(x_train, y_train)
    pred = classifier.predict(x_test)
    return {
        "log_test_rows": len(logs_test), "cluster_silhouette": round(float(silhouette), 4),
        "noisy_alert_accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "noisy_alert_f1": round(float(f1_score(y_test, pred)), 4),
        "noisy_alert_roc_auc": round(float(roc_auc_score(y_test, classifier.predict_proba(x_test)[:, 1])), 4),
        "alert_test_rows": int(len(y_test)),
    }
