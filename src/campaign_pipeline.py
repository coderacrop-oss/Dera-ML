import json
import os
import sys
import time
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from layer5_llm_summarizer import generate_campaign_action_plan
from business_config import CONFIG
from pipeline_logger import log
from crm_export import export_customer_segments_safe, export_segments_safe
from model_registry import ModelRegistry, RegistryError
from uplift_engine import build_serving_context
from atomic_io import atomic_write_text
from uplift_engine import decide_for_customers, _check_features

# ==========================================
# Rule-based fallback scoring (no champion model)
# ==========================================
def _rule_based_score(ctx):
    """Fallback when no champion: assign uplift group from RFM cluster segment.

    Maps cluster_segment → uplift_group heuristic:
      Champions → sure_thing  (will buy anyway)
      At-Risk   → persuadable
      Hibernating (recency > hibernating_recency_days) → lost_cause
      else      → persuadable

    Uses avg_spent as proxy for expected response and still calls build_action_menu()
    for action selection with the margin gate applied.
    """
    from uplift_engine import classify_uplift_group, build_action_menu
    from business_config import CONFIG

    hibernating_days = CONFIG.get("hibernating_recency_days", 90)
    cluster_bundle_map  = getattr(ctx, 'cluster_bundle_map', {}) or {}
    cluster_segment_map = getattr(ctx, 'cluster_segment_map', {}) or {}
    top_bundle_name = ctx.top_bundle_name
    _product_cogs   = ctx.product_cogs
    _default_cogs   = ctx.default_cogs

    rfm = ctx.rfm_df[['customer_id', 'recency', 'kmeans_cluster']].copy()
    rfm['customer_id'] = rfm['customer_id'].astype(str)

    cust = ctx.cust_agg.copy()
    cust['customer_id'] = cust['customer_id'].astype(str)
    cust = cust.merge(rfm, on='customer_id', how='left')

    from uplift_engine import (decide_for_customers as _dfc)
    # Build a minimal mock-model that returns rule-based uplift scores
    import numpy as np

    class _RuleBasedModel:
        """Assigns uplift/prob scores from RFM heuristics; no ML inference."""
        feature_names_ = ctx.features

        @property
        def n_features_(self):
            return len(ctx.features)

        def predict_uplift(self, X):
            # Will be overridden per-customer below — return zeros as placeholder
            return np.zeros(len(X))

        def predict_control_proba(self, X):
            return np.zeros(len(X))

    model = _RuleBasedModel()

    # Score heuristically: use decide_for_customers with injected scores
    rows = []
    for _, row in cust.iterrows():
        seg_name = cluster_segment_map.get(row.get('kmeans_cluster'), '')
        seg_l = seg_name.lower()
        recency = float(row.get('recency', 0))

        if 'champion' in seg_l:
            uplift_score = 0.0
            prob_control = 0.7   # sure_thing threshold is 0.5
        elif 'at-risk' in seg_l or 'churn' in seg_l:
            uplift_score = 0.10  # persuadable
            prob_control = 0.20
        elif 'hibernat' in seg_l or recency > hibernating_days:
            uplift_score = 0.005  # lost_cause
            prob_control = 0.05
        else:
            uplift_score = 0.09   # mid → persuadable
            prob_control = 0.25

        rows.append({'uplift_score': uplift_score, 'prob_control': prob_control})

    score_df = pd.DataFrame(rows)

    # Patch model to return pre-computed scores
    _scores = score_df['uplift_score'].values
    _probs  = score_df['prob_control'].values

    class _PatchedModel(_RuleBasedModel):
        def predict_uplift(self, X):
            return _scores[:len(X)]

        def predict_control_proba(self, X):
            return _probs[:len(X)]

    _, detail = _dfc(_PatchedModel(), ctx)
    return detail


# ==========================================
# Layer 0: CSV Data Loader & Preprocessing
# ==========================================
def load_and_preprocess_csv(file_path):
    print(f"กำลังโหลดและแปลงข้อมูลจากไฟล์ CSV: {file_path}...")
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')

    col_mapping = {
        "CustomerID":   "customer_id",
        "InvoiceDate":  "transaction_date",
        "timestamp":    "transaction_date",
        "Quantity":     "quantity",
        "UnitPrice":    "unit_price",
        "total_price":  "amount",
        "Description":  "product_name",
        "InvoiceNo":    "invoice_id",
    }

    df = df.rename(columns={k: v for k, v in col_mapping.items()
                             if k in df.columns and v not in df.columns})
    df = df.dropna(subset=["customer_id", "transaction_date"])
    # Normalize customer_id: float-formatted IDs (e.g. 17850.0) → clean int string "17850"
    # so they match uplift_engine's _normalize_to_pos_format output.
    try:
        df["customer_id"] = df["customer_id"].astype(float).astype(int).astype(str)
    except (ValueError, TypeError):
        df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors='coerce')

    if "quantity" in df.columns and "unit_price" in df.columns:
        df["amount"] = df["quantity"] * df["unit_price"]
    elif "amount" not in df.columns:
        df["amount"] = 500.0

    df = df[df["amount"] > 0]
    print(f"✅ โหลดสำเร็จ! ข้อมูลดิบทั้งหมด {len(df):,} รายการ จากลูกค้า {df['customer_id'].nunique():,} คน")
    log("L0", "load_complete", rows=len(df), unique_customers=df["customer_id"].nunique(), file=file_path)
    return df

def _segment_urgency(segment_name: str, customer_count: int, avg_monetary: float,
                     trend: str = "stable") -> dict:
    """Compute urgency score and action priority for a segment.

    Score = base_priority × size_weight × trend_weight × value_weight
    Returns dict with: urgency_score (0–100), priority (critical/high/medium/low/monitor),
    action_label, priority_color
    """
    name = segment_name.lower().split(" (")[0]  # strip duplicate suffix like "At-Risk (2)"

    # Base priority per segment type (business logic: who loses most if ignored)
    base = {
        "can't lose them": 95,
        "at-risk":         85,
        "about to sleep":  70,
        "hibernating":     55,
        "need attention":  60,
        "champions":       40,   # monitor, not urgent — already buying
        "loyal customers": 35,
        "potential loyalist": 50,
        "promising":       45,
        "lost":            20,   # low ROI, far gone
    }.get(name, 50)

    # Trend multiplier
    trend_w = {"growing": 1.25, "stable": 1.0, "shrinking": 0.85}.get(trend, 1.0)

    # Size weight: larger segment = more impact (log scale, capped at 1.3x)
    import math
    size_w = min(1.3, 1.0 + math.log1p(customer_count) / 30)

    # Value weight: high AOV segment = higher urgency (1.0 baseline at ฿500)
    value_w = min(1.2, max(0.9, avg_monetary / 500))

    score = min(100, round(base * trend_w * size_w * value_w))

    if score >= 80:
        priority, color, label = "critical", "red",    "ทำก่อน"
    elif score >= 60:
        priority, color, label = "high",     "orange",  "ควรทำ"
    elif score >= 40:
        priority, color, label = "medium",   "yellow",  "ทำได้"
    elif score >= 20:
        priority, color, label = "low",      "green",   "รักษา"
    else:
        priority, color, label = "monitor",  "gray",    "รอก่อน"

    return {
        "urgency_score": score,
        "priority":      priority,
        "priority_color": color,
        "action_label":  label,
    }


def _ibm_segment_name(r_norm: float, f_norm: float, m_norm: float) -> str:
    """Map rank-percentile RFM scores to IBM segment name.

    Inputs are rank-percentile within cluster stats (not raw/max-normalized):
      r_norm: 0–1, inverted (1 = most recent cluster, 0 = oldest)
      f_norm: 0–1 (1 = highest frequency cluster)
      m_norm: 0–1 (1 = highest monetary cluster)

    With rank-percentile normalization, thresholds are relative to k clusters:
      >= 0.6  ≈ top-40% ("high" dimension)
       < 0.4  ≈ bottom-40% ("low" dimension)
      0.4–0.6 ≈ mid-range

    Evaluated most-specific first to avoid ambiguous fallthrough.
    """
    # ── All-round strong ─────────────────────────────────────────────────────
    if r_norm >= 0.6 and f_norm >= 0.6 and m_norm >= 0.6:
        return "Champions"
    # ── High F+M, low R (drifting VIP) ───────────────────────────────────────
    if r_norm < 0.4 and f_norm >= 0.6 and m_norm >= 0.6:
        return "Can't Lose Them"
    if f_norm >= 0.6 and m_norm >= 0.6:
        return "Loyal Customers"
    # ── Low R (infrequent recently) ───────────────────────────────────────────
    if r_norm < 0.4 and f_norm >= 0.4:
        return "At-Risk"
    if r_norm < 0.4:
        return "Hibernating"
    # ── High R (recent) ───────────────────────────────────────────────────────
    if r_norm >= 0.6 and f_norm >= 0.4:
        return "Potential Loyalist"   # recent + decent frequency → build habit
    if r_norm >= 0.6:
        return "Promising"            # recent but low frequency → new customer
    # ── Mid R (0.4–0.6): differentiate by F+M ────────────────────────────────
    if f_norm >= 0.6 or m_norm >= 0.6:
        return "Need Attention"       # still buying well but recency slipping
    if f_norm < 0.4:
        return "About to Sleep"       # low frequency, mid recency → fading
    return "Need Attention"


# ==========================================
# Layer 1: Dynamic k Segmentation + Hybrid Mapping
# ==========================================
def layer1_csv_segmentation(df):
    print("📊 [Layer 1] คำนวณ RFM และหาจำนวนกลุ่ม (k) ที่เหมาะสมที่สุดจากข้อมูลจริง (Dynamic k)...")
    ref_date = df["transaction_date"].max() + pd.Timedelta(days=1)

    df_rfm = df.groupby("customer_id").agg({
        "transaction_date": lambda x: (ref_date - x.max()).days,
        "customer_id": "count",
        "amount": "sum"
    }).rename(columns={
        "transaction_date": "recency",
        "customer_id": "frequency",
        "amount": "monetary"
    }).reset_index().dropna()

    import numpy as _np
    X_log = _np.log1p(df_rfm[["recency", "frequency", "monetary"]].values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    min_k = 3 if len(df_rfm) >= 300 else 2
    best_k = min_k
    best_score = -1
    max_k = min(10, len(df_rfm) - 1)

    for k in range(min_k, max_k + 1):
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_test.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k

    print(f"🎯 Optimal k ที่เลือกได้สำหรับรอบนี้: {best_k} กลุ่ม (Silhouette Score ความคมชัด: {best_score:.3f})")

    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_rfm["cluster"] = final_kmeans.fit_predict(X_scaled)

    cluster_stats = df_rfm.groupby("cluster").agg({
        "recency": "mean",
        "frequency": "mean",
        "monetary": "mean",
        "customer_id": "count"
    }).rename(columns={"customer_id": "count"}).reset_index()

    # Rank-percentile normalization: compare each cluster to other clusters, not to max.
    # Prevents Champions' outlier F/M from making all other clusters look "low".
    max_r = cluster_stats["recency"].max() or 1
    max_f = cluster_stats["frequency"].max() or 1
    max_m = cluster_stats["monetary"].max() or 1
    cluster_stats["rfm_score"] = (
        (1 - cluster_stats["recency"] / max_r) * 0.3 +
        (cluster_stats["frequency"] / max_f) * 0.3 +
        (cluster_stats["monetary"] / max_m) * 0.4
    )
    cluster_stats = cluster_stats.sort_values("rfm_score", ascending=False).reset_index(drop=True)

    if len(cluster_stats) > 1:
        cluster_stats["_r_n"] = 1 - cluster_stats["recency"].rank(pct=True, ascending=True)
        cluster_stats["_f_n"] = cluster_stats["frequency"].rank(pct=True, ascending=True)
        cluster_stats["_m_n"] = cluster_stats["monetary"].rank(pct=True, ascending=True)
    else:
        cluster_stats["_r_n"] = cluster_stats["_f_n"] = cluster_stats["_m_n"] = 0.5

    segments = []
    _used_names: dict = {}
    for rank, (_, row) in enumerate(cluster_stats.iterrows()):
        r_n = row["_r_n"]
        f_n = row["_f_n"]
        m_n = row["_m_n"]
        base_name = _ibm_segment_name(r_n, f_n, m_n)
        count = _used_names.get(base_name, 0)
        seg_name = base_name if count == 0 else f"{base_name} ({count + 1})"
        _used_names[base_name] = count + 1
        urgency = _segment_urgency(seg_name, int(row["count"]), round(row["monetary"], 2))
        segments.append({
            "cluster_id": int(row["cluster"]),
            "segment_name": seg_name,
            "customer_count": int(row["count"]),
            "avg_recency_days": round(row["recency"], 1),
            "avg_frequency": round(row["frequency"], 1),
            "avg_monetary": round(row["monetary"], 2),
            **urgency,
        })

    result = sorted(segments, key=lambda x: x["segment_name"])
    log("L1", "segmentation_complete",
        optimal_k=best_k,
        silhouette_score=round(best_score, 4),
        segments=[s["segment_name"] for s in result],
        unique_customers=int(df_rfm["customer_id"].nunique()))
    return result, df_rfm

# ==========================================
# Shared-context helper: build segments from a pre-built Layer4Context
# (replaces layer1_csv_segmentation when ctx is available)
# ==========================================
def _segments_from_context(ctx):
    """Derive segment list + df_rfm from a Layer4Context built by build_serving_context().

    Using the context's cluster assignments ensures that the segment payload
    and the Layer 4 uplift scores are based on identical cluster membership —
    eliminating the divergence that existed when two separate KMeans runs were used.
    """
    rfm = ctx.rfm_df.rename(columns={"kmeans_cluster": "cluster"}).copy()
    rfm["customer_id"] = rfm["customer_id"].astype(str)

    cluster_stats = rfm.groupby("cluster").agg(
        recency=("recency", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
        count=("customer_id", "count"),
    ).reset_index()

    max_r = cluster_stats["recency"].max() or 1
    max_f = cluster_stats["frequency"].max() or 1
    max_m = cluster_stats["monetary"].max() or 1
    cluster_stats["rfm_score"] = (
        (1 - cluster_stats["recency"] / max_r) * 0.3
        + (cluster_stats["frequency"] / max_f) * 0.3
        + (cluster_stats["monetary"] / max_m) * 0.4
    )
    cluster_stats = cluster_stats.sort_values("rfm_score", ascending=False).reset_index(drop=True)

    if len(cluster_stats) > 1:
        cluster_stats["_r_n"] = 1 - cluster_stats["recency"].rank(pct=True, ascending=True)
        cluster_stats["_f_n"] = cluster_stats["frequency"].rank(pct=True, ascending=True)
        cluster_stats["_m_n"] = cluster_stats["monetary"].rank(pct=True, ascending=True)
    else:
        cluster_stats["_r_n"] = cluster_stats["_f_n"] = cluster_stats["_m_n"] = 0.5

    segments = []
    _used_names: dict = {}
    for rank, (_, row) in enumerate(cluster_stats.iterrows()):
        r_n = row["_r_n"]
        f_n = row["_f_n"]
        m_n = row["_m_n"]
        base_name = _ibm_segment_name(r_n, f_n, m_n)
        count = _used_names.get(base_name, 0)
        seg_name = base_name if count == 0 else f"{base_name} ({count + 1})"
        _used_names[base_name] = count + 1
        urgency = _segment_urgency(seg_name, int(row["count"]), round(row["monetary"], 2))
        segments.append({
            "cluster_id":      int(row["cluster"]),
            "segment_name":    seg_name,
            "customer_count":  int(row["count"]),
            "avg_recency_days": round(row["recency"], 1),
            "avg_frequency":   round(row["frequency"], 1),
            "avg_monetary":    round(row["monetary"], 2),
            **urgency,
        })

    n_clusters = len(segments)
    n_customers = int(rfm["customer_id"].nunique())
    print(f"🎯 [Layer 1 — shared context] {n_clusters} กลุ่ม, {n_customers:,} ลูกค้า (cluster IDs shared with Layer 4)")
    log("L1", "segmentation_complete",
        optimal_k=n_clusters,
        segments=[s["segment_name"] for s in segments],
        unique_customers=n_customers,
        source="shared_layer4_context")

    return sorted(segments, key=lambda x: x["segment_name"]), rfm


# ==========================================
# Layer 2: MBA Co-occurrence จากข้อมูลจริง
# ==========================================
def layer2_csv_mba(df, segments, df_rfm):
    print("🛒 [Layer 2] วิเคราะห์คู่สินค้าซื้อร่วมกัน (Market Basket Analysis) จาก CSV...")
    if "product_name" not in df.columns:
        df["product_name"] = "สินค้าทั่วไป"

    if "invoice_id" not in df.columns:
        # ไม่มี invoice ID จริง → ใช้ customer_id + วันที่เป็น proxy
        # transaction ของลูกค้าคนเดียวกันในวันเดียวกัน = basket เดียวกัน
        date_str = df["transaction_date"].dt.date.astype(str)
        df["invoice_id"] = df["customer_id"].astype(str) + "_" + date_str

    # Limit to top 150 products by frequency to keep co-matrix manageable
    top_products = df["product_name"].value_counts().head(150).index
    df = df[df["product_name"].isin(top_products)]

    basket = df.groupby(["invoice_id", "product_name"])["amount"].count().unstack().fillna(0)
    basket_bool = basket > 0
    # counts need numbers: the product of two True/False tables only says "together at least once" (1), never how many bills
    basket_int = basket_bool.astype(int)

    co_matrix = basket_int.T.dot(basket_int)
    co_mat_np = co_matrix.to_numpy().copy()
    np.fill_diagonal(co_mat_np, 0)
    co_matrix = pd.DataFrame(co_mat_np, index=co_matrix.index, columns=co_matrix.columns)

    def top_pair(matrix):
        """Return (item_a, item_b, count) for highest co-occurrence in matrix."""
        if not matrix.empty and matrix.stack().sum() > 0:
            idx = matrix.stack().idxmax()
            return idx[0], idx[1], int(matrix.loc[idx[0], idx[1]])
        return "สินค้า A", "สินค้า B", 1

    # Global fallback pair
    global_a, global_b, global_count = top_pair(co_matrix)

    enriched_segments = []
    for seg in segments:
        is_champion = "Champion" in seg.get("segment_name", "")

        # Find segment-specific top pair from that segment's customers
        seg_customers = set(
            df_rfm[df_rfm["cluster"] == seg["cluster_id"]]["customer_id"].astype(str)
        )
        seg_invoices = df.loc[
            df["customer_id"].astype(str).isin(seg_customers), "invoice_id"
        ].unique()
        seg_basket = basket_int.loc[basket_int.index.isin(seg_invoices)]

        if len(seg_basket) >= 2:
            seg_co = seg_basket.T.dot(seg_basket)
            seg_co_np = seg_co.to_numpy().copy()
            np.fill_diagonal(seg_co_np, 0)
            seg_co = pd.DataFrame(seg_co_np, index=seg_co.index, columns=seg_co.columns)
            item_a, item_b, co_count = top_pair(seg_co)
        else:
            item_a, item_b, co_count = global_a, global_b, global_count

        if is_champion:
            seg["recommended_promo"] = (
                f"VIP Early Access: [{item_a}] + [{item_b}] "
                f"พร้อมบริการห่อของขวัญฟรีและจัดส่งด่วนก่อนใคร"
            )
            seg["rationale"] = (
                f"[Champions VIP Offer — Co-occurrence: {co_count} txns] "
                f"ไม่ลดราคา รักษา margin สูง เน้นสิทธิพิเศษและประสบการณ์"
            )
        else:
            seg["recommended_promo"] = (
                f"ซื้อ [{item_a}] คู่กับ [{item_b}] ลด 15%"
            )
            seg["rationale"] = (
                f"[Real Data Co-occurrence: {co_count} txns] "
                f"Bundle discount กระตุ้นลูกค้ากลุ่มนี้ให้กลับมาซื้อซ้ำ"
            )

        enriched_segments.append(seg)

    # ── Per-customer preference profile ──────────────────────────────────────
    # สร้าง individual preference สำหรับ insight รายคน
    # (ไม่ส่งผลต่อ campaign decision — export แยกเพื่อ insight เท่านั้น)
    pref_rows = []
    cust_df = df.copy()
    cust_df["customer_id"] = cust_df["customer_id"].astype(str)

    for cid, grp in cust_df.groupby("customer_id"):
        # top-3 products by purchase frequency
        prod_counts = grp["product_name"].value_counts()
        top3 = prod_counts.index.tolist()[:3]
        top3 += [None] * (3 - len(top3))  # pad to length 3

        # personal basket: invoices ที่มีมากกว่า 1 product → personal co-occurrence
        cid_invoices = grp["invoice_id"].unique()
        cid_basket = basket_int.loc[basket_int.index.isin(cid_invoices)]
        personal_pair_a, personal_pair_b = None, None
        personal_pair_count = 0
        if len(cid_basket) >= 1 and cid_basket.shape[1] >= 2:
            cid_co = cid_basket.T.dot(cid_basket)
            cid_co_np = cid_co.to_numpy().copy()
            np.fill_diagonal(cid_co_np, 0)
            cid_co = pd.DataFrame(cid_co_np, index=cid_co.index, columns=cid_co.columns)
            if cid_co.stack().sum() > 0:
                idx = cid_co.stack().idxmax()
                personal_pair_a = idx[0]
                personal_pair_b = idx[1]
                personal_pair_count = int(cid_co.loc[idx[0], idx[1]])

        # recency: last purchase date
        last_date = grp["transaction_date"].max()
        if hasattr(last_date, "date"):
            last_date = str(last_date.date())

        # avg basket size (items per visit)
        avg_basket = round(
            grp.groupby("invoice_id")["product_name"].count().mean(), 1
        )

        pref_rows.append({
            "customer_id":       cid,
            "top_product_1":     top3[0],
            "top_product_2":     top3[1],
            "top_product_3":     top3[2],
            "personal_pair_a":   personal_pair_a,
            "personal_pair_b":   personal_pair_b,
            "personal_pair_times": personal_pair_count if personal_pair_count else None,
            "purchase_freq":     len(grp["invoice_id"].unique()),
            "avg_basket_size":   avg_basket,
            "last_purchased":    last_date,
        })

    pref_df = pd.DataFrame(pref_rows)
    os.makedirs("data/processed", exist_ok=True)
    pref_path = "data/processed/layer2_customer_preferences.csv"
    pref_df.to_csv(pref_path, index=False)
    print(f"👤 [Layer 2] Customer preferences: {len(pref_df)} คน → {pref_path}")

    log("L2", "mba_complete",
        segments_enriched=len(enriched_segments),
        top_global_pair=f"{global_a} + {global_b}",
        top_global_cooccurrence=global_count,
        customer_preferences_exported=len(pref_df))
    return enriched_segments, df_rfm

# ==========================================
# Layer 3 & 4: Timing, Unit Economics & Gate
# ==========================================
def _compute_retention_and_repeat(df_raw, df_rfm, cluster_id):
    """คำนวณ Historical Retention Rate และ Repeat Purchase Rate จากข้อมูลจริงต่อ cluster"""
    # --- Repeat Purchase Rate ---
    seg_rfm = df_rfm[df_rfm["cluster"] == cluster_id]
    repeat_rate = round((seg_rfm["frequency"] > 1).sum() / len(seg_rfm) * 100, 1) if len(seg_rfm) > 0 else 0.0

    # --- Retention Rate (split timeline เป็นสองครึ่ง) ---
    date_min = df_raw["transaction_date"].min()
    date_max = df_raw["transaction_date"].max()
    date_mid = date_min + (date_max - date_min) / 2

    seg_customer_ids = set(seg_rfm["customer_id"].astype(str))
    customers_p1 = set(df_raw[df_raw["transaction_date"] <= date_mid]["customer_id"].astype(str)) & seg_customer_ids
    customers_p2 = set(df_raw[df_raw["transaction_date"] > date_mid]["customer_id"].astype(str)) & seg_customer_ids
    retained = customers_p1 & customers_p2
    retention_rate = round(len(retained) / len(customers_p1) * 100, 1) if customers_p1 else 0.0

    return retention_rate, repeat_rate


def _estimate_rates_from_data(df):
    """
    Estimate COGS rate and discount rate from transaction data.
    - COGS: price-floor method — min_price/avg_price per product ≈ cost/revenue
    - Discount rate: frequency × depth from promo_type distribution
    Falls back to CONFIG defaults if data is insufficient.
    """
    cogs_rate = CONFIG["cogs_rate"]
    discount_rate = CONFIG["default_discount_rate"]
    source = "config_default"

    product_cogs = {}   # per-product COGS dict {product_name: cogs_rate}

    # --- COGS rate: price floor per product ---
    price_col = "unit_price" if "unit_price" in df.columns else None
    if price_col and "product_name" in df.columns:
        df_pos = df[df[price_col] > 0]   # exclude returns / negative prices
        prod = df_pos.groupby("product_name")[price_col].agg(["min", "mean", "count"])
        prod = prod[
            (prod["mean"] > 0) & (prod["min"] > 0) &
            (prod["mean"] > prod["min"]) &
            (prod["count"] >= 5)           # ต้องมีอย่างน้อย 5 observations ต่อ product
        ]
        if len(prod) >= 10:
            # ใช้ 10th percentile แทน min เพื่อลด noise จาก outlier price
            price_p10 = df_pos.groupby("product_name")[price_col].quantile(0.10)
            prod["p10"] = prod.index.map(price_p10)
            prod["cogs_est"] = (prod["p10"] / prod["mean"]).clip(0.10, 0.90)
            cost_ratio = prod["cogs_est"].median()
            if 0.20 < cost_ratio < 0.80:
                cogs_rate = round(float(cost_ratio), 3)
                source = "data_driven"
                product_cogs = prod["cogs_est"].round(3).to_dict()
                print(f"💡 [Rates] Per-product COGS estimated for {len(product_cogs):,} products "
                      f"(range {prod['cogs_est'].min():.1%}–{prod['cogs_est'].max():.1%})")

    # --- Discount rate: promo frequency × CONFIG depth ---
    if "promo_type" in df.columns:
        disc_freq = (df["promo_type"] != "none").mean()
        estimated = round(disc_freq * CONFIG["default_discount_rate"], 4)
        if 0.005 < estimated < 0.40:
            discount_rate = estimated
            source = "data_driven"

    print(f"💡 [Rates] COGS={cogs_rate:.1%}  Discount={discount_rate:.1%}  source={source}")
    return cogs_rate, discount_rate, product_cogs


def layer3_and_4_processing(segments, df_raw, df_rfm):
    print("⏰ [Layer 3 & 4] คำนวณตารางเวลา, Unit Economics และ Post-Campaign Forecast...")
    updated_segments = []
    total_profit = 0

    # Estimate rates from data (falls back to CONFIG if insufficient)
    estimated_cogs, estimated_discount, product_cogs = _estimate_rates_from_data(df_raw)

    for seg in segments:
        aov = seg["avg_monetary"]
        cust_count = seg["customer_count"]
        cogs_rate = estimated_cogs
        # Champions: VIP service (no % discount) — use vip_points cost rate
        # Other segments: use data-estimated discount rate
        is_champion = "Champion" in seg.get("segment_name", "")
        discount_rate = (
            CONFIG["promo_cost_rates"]["vip_points"] if is_champion
            else estimated_discount
        )

        net_profit_per_head = aov * (1 - cogs_rate - discount_rate)
        total_seg_profit = net_profit_per_head * cust_count

        # คำนวณ metrics จากข้อมูลจริง
        retention_rate, repeat_rate = _compute_retention_and_repeat(df_raw, df_rfm, seg["cluster_id"])

        # Projected post-campaign: Champions ได้ VIP (uplift น้อย), At-Risk ได้ discount (uplift มากกว่า)
        is_champion = "Champion" in seg["segment_name"]
        retention_uplift = CONFIG["retention_uplift_champion"] if is_champion else CONFIG["retention_uplift_atrisk"]
        repeat_uplift    = CONFIG["repeat_uplift_champion"]    if is_champion else CONFIG["repeat_uplift_atrisk"]
        projected_retention = round(min(retention_rate * (1 + retention_uplift), 98.0), 1)
        projected_repeat = round(min(repeat_rate * (1 + repeat_uplift), 98.0), 1)

        seg["send_schedule"] = "ทุกวันศุกร์ เวลา 17:00 น."
        seg["campaign_duration"] = "7 วัน"
        seg["unit_economics"] = {
            "net_profit_per_head": round(net_profit_per_head, 2),
            "total_projected_profit": round(total_seg_profit, 2),
            "financial_status": "PROFITABLE" if net_profit_per_head > 0 else "LOSS ALERT"
        }
        seg["historical_metrics"] = {
            "retention_rate_pct": retention_rate,
            "repeat_purchase_rate_pct": repeat_rate,
            "data_period": f"{df_raw['transaction_date'].min().date()} ถึง {df_raw['transaction_date'].max().date()}"
        }
        seg["post_campaign_forecast"] = {
            "projected_retention_rate_pct": projected_retention,
            "projected_repeat_purchase_rate_pct": projected_repeat,
            "retention_uplift_assumption_pct": round(retention_uplift * 100),
            "repeat_uplift_assumption_pct": round(repeat_uplift * 100)
        }

        total_profit += total_seg_profit
        updated_segments.append(seg)

    payload = {
        "campaign_name": "CSV Data Driven Retail Campaign",
        "planning_month": "พฤศจิกายน 2026",
        "target_segments": updated_segments,
        "gate_status": "PASSED",
        "total_campaign_projected_profit": f"{total_profit:,.2f} บาท"
    }
    log("L3", "unit_economics_complete",
        segments_processed=len(updated_segments),
        total_projected_profit=round(total_profit, 2),
        cogs_rate=estimated_cogs,
        discount_rate=estimated_discount,
        gate_status="PASSED")
    return payload, product_cogs, estimated_cogs


_NEW_OUTPUT_COLUMNS = ["action_key", "cost_source", "action_group", "profit_basis", "schema_version"]


def _reconcile_promo_advisor(detail):
    """Keep action_key / cost_source / action_group / profit_basis truthful after the engine's decision was
    changed downstream: the Promo Advisor may replace the action with the shop's own promo (operator_promo /
    operator), and the Promo Advisor or the budget knapsack may turn a row into "Do Nothing ..." (no key,
    cost_source none).  Whatever made a row do nothing, it claims no promotion."""
    if "action_key" not in detail.columns:
        return detail
    detail = detail.copy()
    if "final_source" in detail.columns:
        op = detail["final_source"] == "operator"
        detail.loc[op, "action_key"] = "operator_promo"
        detail.loc[op, "cost_source"] = "operator"
        detail.loc[op, "action_group"] = "other"
        detail.loc[op, "profit_basis"] = None
        nothing = detail["final_source"] == "none"
    else:
        nothing = pd.Series(False, index=detail.index)
    action = detail["recommended_action"].fillna("").astype(str)
    nothing = nothing | action.str.startswith("Do Nothing") | action.str.startswith("Zero-Cost")
    for col, value in (("action_key", None), ("action_group", None), ("profit_basis", None), ("cost_source", "none")):
        detail.loc[nothing, col] = value
    return detail


def _new_columns_last(detail):
    """Existing output columns keep their names and order; the new ones go at the very end."""
    from business_config import OUTPUT_SCHEMA_VERSION
    if "schema_version" not in detail.columns:
        detail = detail.assign(schema_version=OUTPUT_SCHEMA_VERSION)
    last = [c for c in _NEW_OUTPUT_COLUMNS if c in detail.columns]
    return detail[[c for c in detail.columns if c not in last] + last]


def _layer4_view(detail):
    """The per-customer columns Layer 4 hands to merge_layer4_into_payload (broadcast list, segments)."""
    keep_cols = ['customer_id', 'recommended_action', 'expected_profit',
                 'guardrail_blocked', 'should_send', 'send_window']
    if 'cluster' in detail.columns:
        keep_cols.insert(1, 'cluster')
    if 'treatment_type' in detail.columns:
        keep_cols.append('treatment_type')
    keep_cols += [c for c in _NEW_OUTPUT_COLUMNS if c in detail.columns]
    return detail[keep_cols]


def merge_layer4_into_payload(payload: dict, layer4_df, df_rfm) -> dict:
    """
    Merge Layer 4 per-customer uplift decisions into segment plan.
    Updates each segment with:
      - broadcast_count:  customers Layer 4 approved to send (should_send=1)
      - suppressed_count: customers blocked by CFO guardrail (should_send=0)
      - suppression_rate: % ที่ถูก block
      - l4_action_breakdown: สัดส่วน recommended_action ในกลุ่ม
    Also exports broadcast_list.csv for automation tools.
    """
    import os
    import pandas as pd

    if layer4_df is None or layer4_df.empty:
        return payload

    # layer4_df has 'cluster' column — map to segment cluster_id
    layer4_df = layer4_df.copy()
    layer4_df['customer_id'] = layer4_df['customer_id'].astype(str)

    # Attach cluster label from df_rfm if not already present
    if 'cluster' not in layer4_df.columns and df_rfm is not None:
        rfm_map = df_rfm[['customer_id', 'cluster']].copy()
        rfm_map['customer_id'] = rfm_map['customer_id'].astype(str)
        layer4_df = layer4_df.merge(rfm_map, on='customer_id', how='left')

    updated_segments = []
    broadcast_rows = []
    total_suppressed = 0
    total_broadcast = 0

    for seg in payload["target_segments"]:
        cluster_id = seg["cluster_id"]
        seg_df = layer4_df[layer4_df['cluster'] == cluster_id]

        if seg_df.empty:
            # Layer 4 didn't cover this cluster — keep all customers
            seg["broadcast_count"]  = seg["customer_count"]
            seg["suppressed_count"] = 0
            seg["suppression_rate"] = 0.0
            seg["l4_available"]     = False
            total_broadcast += seg["customer_count"]
            updated_segments.append(seg)
            continue

        should_send  = seg_df[seg_df['should_send'] == 1]
        suppressed   = seg_df[seg_df['should_send'] == 0]
        bc_count     = len(should_send)
        sup_count    = len(suppressed)
        sup_rate     = round(sup_count / len(seg_df) * 100, 1) if len(seg_df) > 0 else 0.0

        # Action breakdown
        action_counts = seg_df['recommended_action'].value_counts().to_dict()

        # Recalculate projected profit using only broadcast customers' expected_profit
        l4_total_profit = round(should_send['expected_profit'].sum(), 2)

        seg["broadcast_count"]    = bc_count
        seg["suppressed_count"]   = sup_count
        seg["suppression_rate"]   = sup_rate
        seg["l4_action_breakdown"] = action_counts
        # same counts keyed by the stable action_key ('none' = Do Nothing / Zero-Cost) — additive field
        seg["l4_action_key_breakdown"] = (
            seg_df['action_key'].astype(object).where(seg_df['action_key'].notna(), 'none').value_counts().to_dict()
            if 'action_key' in seg_df.columns else None)
        seg["l4_available"]       = True

        # Override projected profit with Layer 4's more accurate estimate
        seg["unit_economics"]["l4_total_projected_profit"] = l4_total_profit
        seg["unit_economics"]["l4_profit_note"] = (
            f"Layer 4 approved {bc_count}/{seg['customer_count']} customers "
            f"({100-sup_rate:.0f}% of segment)"
        )

        total_broadcast  += bc_count
        total_suppressed += sup_count

        # Build broadcast rows for this segment
        send_date = pd.Timestamp("today").normalize() + pd.offsets.Week(weekday=4)  # next Friday
        for _, row in should_send.iterrows():
            send_window = row.get("send_window", "Peak Window")
            # send_time: 30 min before the target window opens
            # Off-Peak customers → notify just before off-peak window so they plan to come
            # Peak customers     → notify just before peak window as a reminder
            send_time = "Before Off-Peak Window" if "Off-Peak" in str(send_window) else "Before Peak Window"
            broadcast_rows.append({
                "customer_id":         row["customer_id"],
                "segment_name":        seg["segment_name"],
                "recommended_action":  row["recommended_action"],
                "offer":               seg.get("recommended_promo", "-"),
                "expected_profit":     row["expected_profit"],
                "promo_cost":          row.get("promo_cost", 0.0),
                "send_date":           send_date.date(),
                "send_window":         send_window,
                "send_time_hint":      send_time,
                "channel":             "Email/SMS",
                "should_send":         1,
                "action_key":          row.get("action_key"),
                "cost_source":         row.get("cost_source"),
                "action_group":        row.get("action_group"),
                "profit_basis":        row.get("profit_basis"),
                "schema_version":      row.get("schema_version"),
            })

        updated_segments.append(seg)

    payload["target_segments"] = updated_segments
    payload["broadcast_summary"] = {
        "total_broadcast": total_broadcast,
        "total_suppressed": total_suppressed,
        "suppression_rate_pct": round(total_suppressed / (total_broadcast + total_suppressed) * 100, 1)
        if (total_broadcast + total_suppressed) > 0 else 0.0,
    }

    # Export broadcast list CSV
    if broadcast_rows:
        os.makedirs("data/processed", exist_ok=True)
        broadcast_path = "data/processed/broadcast_list.csv"
        pd.DataFrame(broadcast_rows).to_csv(broadcast_path, index=False)
        print(f"📤 [Broadcast] Exported {total_broadcast} customers → {broadcast_path}")
        print(f"🚫 [Broadcast] Suppressed {total_suppressed} customers (CFO guardrail / low uplift)")

    return payload

# ==========================================
# Layer 4 Serving: score every customer with the approved champion
# ==========================================
def run_layer4(pos_file_path="data/transactions.csv",
               promo_response_path="data/marketing_campaign.csv",
               product_cogs_dict=None,
               default_cogs=None,
               models_root=None,
               ctx=None,
               campaign_budget=None,
               promo_spec=None):
    """Serving: score every customer with the approved champion. Never trains, never writes the registry.

    Pass a pre-built Layer4Context as `ctx` to avoid re-running L0-L3 when the caller
    has already built the context for cluster assignment.

    Parameters
    ----------
    campaign_budget : float or None
        Optional spend ceiling (same currency as promo_cost).  When set, the
        broadcast list is filtered via the greedy knapsack so that total
        promo_cost stays within this budget.  Pass None (default) to send to
        all CFO-approved customers without a budget constraint.
    promo_spec : PromoSpec or None
        Optional operator-planned promotion.  When provided, the Promo Advisor
        runs after Layer 4 decisions and produces a merged broadcast list that
        picks the higher-profit action per customer (ML vs operator promo).
        Results saved to data/processed/promo_advisor_output.csv and
        data/processed/promo_advisor_report.json.
    """
    print("=== กำลังประมวลผล Layer 4: Campaign Engine (Serving — ใช้ Champion ที่อนุมัติแล้ว) ===")

    if ctx is None and not os.path.exists(pos_file_path):
        print(f"❌ ไม่พบไฟล์ข้อมูล: {pos_file_path}")
        return

    from uplift_engine import REGISTRY_PATH
    registry = ModelRegistry(models_root or os.path.dirname(REGISTRY_PATH))
    champion, ref = registry.load_current()   # fails before any output if there is no usable champion

    if ctx is None:
        ctx = build_serving_context(pos_file_path, promo_response_path, product_cogs_dict, default_cogs)
        # Build cluster_bundle_map from Layer 2 so actions use per-cluster bundles
        if not ctx.cluster_bundle_map:
            _df_raw = load_and_preprocess_csv(pos_file_path)
            _segs, _df_rfm = _segments_from_context(ctx)
            _segs_l2, _ = layer2_csv_mba(_df_raw, _segs, _df_rfm)
            ctx.cluster_bundle_map = {
                seg["cluster_id"]: seg.get("recommended_promo", "")
                for seg in _segs_l2
            }
            ctx.cluster_segment_map = {
                seg["cluster_id"]: seg.get("segment_name", "")
                for seg in _segs_l2
            }
    _check_features(champion, ctx.features)

    # ── P0: Feature drift detection ───────────────────────────────────────────
    baseline = getattr(champion, "feature_baseline_stats", {})
    if baseline:
        try:
            from drift_detector import check_drift, print_drift_report
            drift_report = check_drift(baseline, ctx.cust_agg, ctx.features)
            print_drift_report(drift_report)
            log("L4", "drift_check", **{k: v for k, v in drift_report.items()
                                        if k not in ("per_feature", "recommendation")})
            # Save drift report alongside model_used.json
            os.makedirs("data/processed", exist_ok=True)
            atomic_write_text(
                "data/processed/drift_report.json",
                json.dumps(drift_report, indent=2, ensure_ascii=False),
            )
        except Exception as _e:
            print(f"⚠️  [Drift] Could not run drift check: {_e}")
    else:
        print("ℹ️  [Drift] Champion has no baseline stats (trained before P0 upgrade) — skipping drift check")

    _, detail = decide_for_customers(champion, ctx)

    # ── Budget knapsack (optional) ────────────────────────────────────────────
    if campaign_budget is not None and campaign_budget > 0:
        from budget_optimizer import select_within_budget, print_budget_report
        detail = select_within_budget(detail, budget=campaign_budget)
        # Suppress customers excluded by budget constraint
        detail.loc[
            (detail["should_send"] == 1) & (~detail["selected_for_budget"].fillna(False)),
            ["should_send", "recommended_action"]
        ] = [0, "Do Nothing (Over Budget)"]
        print_budget_report(detail, campaign_budget)

    # ── Promo Advisor (optional) ──────────────────────────────────────────────
    _promo_report = None
    if promo_spec is not None:
        try:
            from promo_advisor import promo_advisor, print_promo_report
            _cogs  = getattr(ctx, "cogs_rate",  None) or 0.50
            _floor = 0.15  # min_allowable_net_margin
            _promo_df, _promo_report = promo_advisor(
                ml_output=detail,
                cust_agg=ctx.cust_agg,
                spec=promo_spec,
                cogs_rate=_cogs,
                min_margin=_floor,
                history_path="data/processed/promo_history.jsonl",
            )
            print_promo_report(_promo_report)
            os.makedirs("data/processed", exist_ok=True)
            _promo_df.to_csv("data/processed/promo_advisor_output.csv", index=False)
            atomic_write_text(
                "data/processed/promo_advisor_report.json",
                json.dumps(_promo_report, indent=2, ensure_ascii=False),
            )
            # Replace detail with promo-advisor resolved output for final broadcast list
            # Map final_should_send / final_action back to standard columns
            _cols_map = {
                "final_should_send": "should_send",
                "final_action":      "recommended_action",
                "final_profit":      "expected_profit",
                "final_promo_cost":  "promo_cost",
            }
            for src, dst in _cols_map.items():
                if src in _promo_df.columns:
                    detail = detail.merge(
                        _promo_df[["customer_id", src]].drop_duplicates("customer_id"),
                        on="customer_id", how="left",
                    )
                    detail[dst] = detail[src].combine_first(detail[dst])
                    detail.drop(columns=[src], inplace=True)
            if "advisor_reason" in _promo_df.columns:
                detail = detail.merge(
                    _promo_df[["customer_id", "advisor_reason",
                               "final_source", "conflict_type"]].drop_duplicates("customer_id"),
                    on="customer_id", how="left",
                )
        except Exception as _pa_exc:
            print(f"⚠️  [PromoAdvisor] ข้ามเพราะ: {_pa_exc}")

    detail = _reconcile_promo_advisor(detail)
    total_cust = len(detail)
    print(f"\n🎯 การตัดสินใจ Layer 4 ({total_cust:,} ลูกค้า) โดย Champion {ref.version} [{ref.provenance}]:")
    for action, cnt in detail['recommended_action'].value_counts().items():
        icon = "✓" if not action.startswith("Do Nothing") else "✗"
        print(f"   {icon} {action:<45} {cnt:>5,} คน ({cnt / total_cust * 100:.1f}%)")
    print("-------------------------------------------------------")

    os.makedirs("data/processed", exist_ok=True)
    detail = _new_columns_last(detail)
    detail.to_csv("data/processed/final_campaign_engine_output.csv", index=False)
    atomic_write_text("data/processed/model_used.json", json.dumps(
        {"champion_version": ref.version, "sha256": ref.sha256, "provenance": ref.provenance}, indent=2))
    _treatment_breakdown = detail['treatment_type'].value_counts().to_dict() if 'treatment_type' in detail.columns else {}
    log("L4", "l4_serving_complete", champion_version=ref.version, customers=total_cust,
        send_count=int(detail['should_send'].sum()),
        treatment_breakdown=_treatment_breakdown)

    # ── P1: Retraining trigger check ──────────────────────────────────────────
    try:
        from retrain_trigger import assess_retrain
        trigger = assess_retrain(models_root or os.path.dirname(REGISTRY_PATH))
        if trigger["needs_retrain"]:
            print("\n⚠️  [Retrain] Retraining recommended:")
            for r in trigger["reasons"]:
                print(f"   ❌ {r}")
            print("   → Run: python src/train_champion.py --store-data STORE:data/file.csv")
        elif trigger["warnings"]:
            for w in trigger["warnings"]:
                print(f"   ℹ️  [Retrain] {w}")
    except Exception as _e:
        log("L4", "retrain_trigger_error", error=str(_e), level="WARN")

    # ── P2 #5: Shadow mode — score latest pending candidate without affecting decisions ──
    try:
        from candidates import list_pending_candidates, candidate_model_bytes as _cand_bytes
        _root = models_root or os.path.dirname(REGISTRY_PATH)
        pending = list_pending_candidates(_root)
        if pending:
            _cid     = pending[0]["candidate_id"]
            _shadow  = pickle.loads(_cand_bytes(_root, _cid))
            _s_feats = getattr(_shadow, "feature_names_", ctx.features)
            if list(_s_feats) == list(ctx.features):
                _, _sd = decide_for_customers(_shadow, ctx)
                _s_profit = float(_sd["expected_profit"].sum())
                _c_profit = float(detail["expected_profit"].sum())
                _diff_pct = round((_s_profit - _c_profit) / abs(_c_profit) * 100, 2) if _c_profit else None
                _shadow_rpt = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "champion_version":      ref.version,
                    "shadow_candidate":      _cid,
                    "champion_send_rate_pct": round(float(detail["should_send"].mean() * 100), 2),
                    "shadow_send_rate_pct":   round(float(_sd["should_send"].mean() * 100), 2),
                    "champion_profit":        round(_c_profit, 2),
                    "shadow_profit":          round(_s_profit, 2),
                    "profit_diff_pct":        _diff_pct,
                }
                atomic_write_text("data/processed/shadow_report.json",
                                  json.dumps(_shadow_rpt, indent=2, ensure_ascii=False))
                _diff_str = f"{_diff_pct:+.1f}%" if _diff_pct is not None else "N/A"
                print(f"\n👥 [Shadow] Candidate {_cid}: "
                      f"send={_shadow_rpt['shadow_send_rate_pct']:.1f}%  "
                      f"profit={_s_profit:,.2f}  (vs champion: {_diff_str})")
                log("L4", "shadow_run", **_shadow_rpt)
            else:
                print(f"\n👥 [Shadow] Candidate {_cid} skipped — feature mismatch")
    except Exception as _e:
        log("L4", "shadow_run_error", error=str(_e), level="WARN")

    # ── P2 #8: Business metric monitoring ─────────────────────────────────────
    try:
        from business_monitor import record_snapshot
        snap = record_snapshot(detail, ref.version)
        print(f"📊 [Monitor] send={snap.get('send_rate_pct','?')}%  "
              f"avg_profit={snap.get('avg_expected_profit','?')}  "
              f"do_nothing={snap.get('do_nothing_pct','?')}%  "
              f"[history → data/processed/business_metrics_history.csv]")
    except Exception as _e:
        log("L4", "business_monitor_error", error=str(_e), level="WARN")

    return _layer4_view(detail)


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(add_help=False)
    _parser.add_argument("csv_file", nargs="?", default="data/transactions.csv")
    _parser.add_argument("--promo-response", default=None,
                         help="Path to labels CSV (customer_id, is_treated, post_promo_visit)")
    _cli, _ = _parser.parse_known_args()
    csv_file = _cli.csv_file
    _cli_promo_response = _cli.promo_response
    print("🚀 เริ่มต้นรัน CSV MLOps Campaign Pipeline...\n")
    log("L0", "pipeline_start", csv_file=csv_file)
    _pipeline_start = time.monotonic()
    try:
        from uplift_engine import REGISTRY_PATH as _REGISTRY_PATH

        _t0 = time.monotonic()
        df_raw = load_and_preprocess_csv(csv_file)
        log("L0", "layer_complete", duration_ms=round((time.monotonic() - _t0) * 1000), rows_out=len(df_raw))

        # ── Resolve promo label source (moved up so build_serving_context gets real labels) ──
        # Priority: explicit CLI arg > POS file has treatment cols > marketing_campaign companion > proxy
        _pos_abs = os.path.abspath(csv_file)
        _pos_dir = os.path.dirname(_pos_abs)
        _pos_name = os.path.basename(_pos_abs).lower()
        _header = pd.read_csv(_pos_abs, nrows=0).columns.tolist()
        if _cli_promo_response and os.path.exists(_cli_promo_response):
            promo_path = os.path.abspath(_cli_promo_response)
            _promo_source = "explicit_arg"
        elif 'is_treated' in _header and 'post_promo_visit' in _header:
            promo_path = _pos_abs
            _promo_source = "pos_file_direct"
        elif 'marketing_campaign' in _pos_name:
            _candidate = os.path.join(_pos_dir, "marketing_campaign.csv")
            promo_path = _candidate if os.path.exists(_candidate) else None
            _promo_source = "marketing_campaign_companion" if promo_path else "rfm_proxy_fallback"
        else:
            promo_path = None
            _promo_source = "rfm_proxy_fallback"
        log("L4", "promo_path_routing",
            promo_source=_promo_source, promo_path=str(promo_path), csv_file=csv_file)

        # ── Build shared Layer4Context (runs L0-L3 internally, drives cluster IDs for all layers) ──
        # By sharing this context with the serving step we guarantee that the cluster memberships
        # used for the segment payload are identical to those used as ML features — one KMeans, not two.
        print("\n🔗 [Shared Context] กำลังสร้าง Layer4Context (L1-L3 ร่วมกัน)...")
        _t0 = time.monotonic()
        _shared_ctx = build_serving_context(csv_file, promo_path)
        log("L0", "shared_context_built", duration_ms=round((time.monotonic() - _t0) * 1000),
            customers=len(_shared_ctx.cust_agg), label_mode=_shared_ctx.label_mode)

        _t0 = time.monotonic()
        _rows_in = len(df_raw)
        segments, df_rfm = _segments_from_context(_shared_ctx)
        log("L1", "layer_complete", duration_ms=round((time.monotonic() - _t0) * 1000),
            rows_in=_rows_in, rows_out=int(df_rfm["customer_id"].nunique()))

        _t0 = time.monotonic()
        _rows_in = int(df_rfm["customer_id"].nunique())
        segments, df_rfm = layer2_csv_mba(df_raw, segments, df_rfm)
        log("L2", "layer_complete", duration_ms=round((time.monotonic() - _t0) * 1000),
            rows_in=_rows_in, rows_out=len(segments))

        # Patch cluster maps into shared context so Layer 4 uses per-cluster bundles
        _shared_ctx.cluster_bundle_map = {
            seg["cluster_id"]: seg.get("recommended_promo", "")
            for seg in segments
        }
        _shared_ctx.cluster_segment_map = {
            seg["cluster_id"]: seg.get("segment_name", "")
            for seg in segments
        }

        _t0 = time.monotonic()
        validated_payload, product_cogs, default_cogs = layer3_and_4_processing(segments, df_raw, df_rfm)
        log("L3", "layer_complete", duration_ms=round((time.monotonic() - _t0) * 1000),
            rows_in=len(segments), rows_out=len(validated_payload["target_segments"]))

        # Patch per-product COGS into the shared context so Layer 4 profit estimates
        # use the same data-driven rates computed by Layer 3.
        _shared_ctx.product_cogs = product_cogs
        _shared_ctx.default_cogs = default_cogs

        print("\n🔍 [Layer 4] รัน LightGBM Uplift + CFO Guardrail...")
        _t0 = time.monotonic()

        # ── Layer 4 Serving (inlined from run_layer4) ─────────────────────────
        print("=== กำลังประมวลผล Layer 4: Campaign Engine (Serving — ใช้ Champion ที่อนุมัติแล้ว) ===")
        registry = ModelRegistry(os.path.dirname(_REGISTRY_PATH))
        _use_rule_based = False
        try:
            champion, ref = registry.load_current()
            _check_features(champion, _shared_ctx.features)
        except RegistryError as _reg_exc:
            if "no champion" in str(_reg_exc).lower() or "train_champion" in str(_reg_exc).lower():
                print("[Fallback] ไม่มี champion model — ใช้ rule-based scoring จาก RFM cluster")
                log("L4", "rule_based_fallback", reason=str(_reg_exc), level="WARN")
                _use_rule_based = True
                champion = None
                ref = None
            else:
                raise

        # ── P0: Feature drift detection ───────────────────────────────────────────
        if not _use_rule_based:
            baseline = getattr(champion, "feature_baseline_stats", {})
            if baseline:
                try:
                    from drift_detector import check_drift, print_drift_report
                    drift_report = check_drift(baseline, _shared_ctx.cust_agg, _shared_ctx.features)
                    print_drift_report(drift_report)
                    log("L4", "drift_check", **{k: v for k, v in drift_report.items()
                                                if k not in ("per_feature", "recommendation")})
                    # Save drift report alongside model_used.json
                    os.makedirs("data/processed", exist_ok=True)
                    atomic_write_text(
                        "data/processed/drift_report.json",
                        json.dumps(drift_report, indent=2, ensure_ascii=False),
                    )
                except Exception as _e:
                    print(f"⚠️  [Drift] Could not run drift check: {_e}")
            else:
                print("ℹ️  [Drift] Champion has no baseline stats (trained before P0 upgrade) — skipping drift check")

        if _use_rule_based:
            detail = _rule_based_score(_shared_ctx)
        else:
            _, detail = decide_for_customers(champion, _shared_ctx)

        total_cust = len(detail)
        if _use_rule_based:
            print(f"\n🎯 การตัดสินใจ Layer 4 ({total_cust:,} ลูกค้า) — Rule-Based Fallback (RFM Cluster):")
        else:
            print(f"\n🎯 การตัดสินใจ Layer 4 ({total_cust:,} ลูกค้า) โดย Champion {ref.version} [{ref.provenance}]:")
        for action, cnt in detail['recommended_action'].value_counts().items():
            icon = "✅" if not action.startswith("Do Nothing") else "⛔"
            print(f"   {icon} {action:<45} {cnt:>5,} คน ({cnt / total_cust * 100:.1f}%)")
        print("-------------------------------------------------------")

        os.makedirs("data/processed", exist_ok=True)
        detail = _new_columns_last(detail)
        detail.to_csv("data/processed/final_campaign_engine_output.csv", index=False)
        if not _use_rule_based:
            atomic_write_text("data/processed/model_used.json", json.dumps(
                {"champion_version": ref.version, "sha256": ref.sha256, "provenance": ref.provenance}, indent=2))
        _treatment_breakdown = detail['treatment_type'].value_counts().to_dict() if 'treatment_type' in detail.columns else {}
        if _use_rule_based:
            log("L4", "l4_serving_complete", champion_version="rule_based", customers=total_cust,
                send_count=int(detail['should_send'].sum()),
                treatment_breakdown=_treatment_breakdown)
        else:
            log("L4", "l4_serving_complete", champion_version=ref.version, customers=total_cust,
                send_count=int(detail['should_send'].sum()),
                treatment_breakdown=_treatment_breakdown)

        # ── P1: Retraining trigger check ──────────────────────────────────────────
        try:
            from retrain_trigger import assess_retrain
            trigger = assess_retrain(os.path.dirname(_REGISTRY_PATH))
            if trigger["needs_retrain"]:
                print("\n⚠️  [Retrain] Retraining recommended:")
                for r in trigger["reasons"]:
                    print(f"   ❌ {r}")
                print("   → Run: python src/train_champion.py --store-data STORE:data/file.csv")
            elif trigger["warnings"]:
                for w in trigger["warnings"]:
                    print(f"   ℹ️  [Retrain] {w}")
        except Exception as _e:
            log("L4", "retrain_trigger_error", error=str(_e), level="WARN")

        # ── P2 #5: Shadow mode — score latest pending candidate without affecting decisions ──
        if not _use_rule_based:
            try:
                from candidates import list_pending_candidates, candidate_model_bytes as _cand_bytes
                _root = os.path.dirname(_REGISTRY_PATH)
                pending = list_pending_candidates(_root)
                if pending:
                    _cid     = pending[0]["candidate_id"]
                    _shadow  = pickle.loads(_cand_bytes(_root, _cid))
                    _s_feats = getattr(_shadow, "feature_names_", _shared_ctx.features)
                    if list(_s_feats) == list(_shared_ctx.features):
                        _, _sd = decide_for_customers(_shadow, _shared_ctx)
                        _s_profit = float(_sd["expected_profit"].sum())
                        _c_profit = float(detail["expected_profit"].sum())
                        _diff_pct = round((_s_profit - _c_profit) / abs(_c_profit) * 100, 2) if _c_profit else None
                        _shadow_rpt = {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "champion_version":      ref.version,
                            "shadow_candidate":      _cid,
                            "champion_send_rate_pct": round(float(detail["should_send"].mean() * 100), 2),
                            "shadow_send_rate_pct":   round(float(_sd["should_send"].mean() * 100), 2),
                            "champion_profit":        round(_c_profit, 2),
                            "shadow_profit":          round(_s_profit, 2),
                            "profit_diff_pct":        _diff_pct,
                        }
                        atomic_write_text("data/processed/shadow_report.json",
                                          json.dumps(_shadow_rpt, indent=2, ensure_ascii=False))
                        _diff_str = f"{_diff_pct:+.1f}%" if _diff_pct is not None else "N/A"
                        print(f"\n👥 [Shadow] Candidate {_cid}: "
                              f"send={_shadow_rpt['shadow_send_rate_pct']:.1f}%  "
                              f"profit={_s_profit:,.2f}  (vs champion: {_diff_str})")
                        log("L4", "shadow_run", **_shadow_rpt)
                    else:
                        print(f"\n👥 [Shadow] Candidate {_cid} skipped — feature mismatch")
            except Exception as _e:
                log("L4", "shadow_run_error", error=str(_e), level="WARN")

        layer4_df = _layer4_view(detail)
        # ── End Layer 4 Serving ───────────────────────────────────────────────

        _l4_rows = len(layer4_df) if layer4_df is not None else 0
        log("L4", "layer_complete", duration_ms=round((time.monotonic() - _t0) * 1000),
            rows_out=_l4_rows,
            broadcast_count=int((layer4_df["should_send"] == 1).sum()) if layer4_df is not None else 0,
            suppressed_count=int((layer4_df["should_send"] == 0).sum()) if layer4_df is not None else 0)

        # customer -> segment, for the CRM API; only after Layer 4 succeeded, so an aborted serving run
        # (no/incompatible champion) never leaves the API with a half-updated set of files
        export_customer_segments_safe(df_rfm, segments)
        validated_payload = merge_layer4_into_payload(validated_payload, layer4_df, df_rfm)
        export_segments_safe(validated_payload)  # for the CRM API; never aborts the pipeline

        print("\n🤖 [Layer 5] ส่งข้อมูลให้ Local LLM สร้าง Action Plan...")
        print("\n" + "="*50)
        _t0 = time.monotonic()
        final_plan = generate_campaign_action_plan(validated_payload)
        log("L5", "layer_complete", duration_ms=round((time.monotonic() - _t0) * 1000))

        log("L0", "pipeline_end", status="SUCCESS",
            total_duration_ms=round((time.monotonic() - _pipeline_start) * 1000))
    except RegistryError as exc:
        log("L0", "pipeline_error", level="ERROR", exc="RegistryError: " + str(exc))
        print(f"❌ ใช้งาน Champion ไม่ได้: {exc}")
        sys.exit(1)
    except FileNotFoundError as exc:
        log("L0", "pipeline_error", level="ERROR", exc=str(exc), csv_file=csv_file)
        print(f"⚠️ ไม่พบไฟล์ {csv_file} กรุณานำไฟล์ CSV วางไว้ในโฟลเดอร์ data/ หรือปรับ path ให้ถูกต้องครับ")
    except Exception as exc:
        log("L0", "pipeline_error", level="ERROR", exc=type(exc).__name__ + ": " + str(exc))
        raise
