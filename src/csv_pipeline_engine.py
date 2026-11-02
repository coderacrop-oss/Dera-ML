import json
import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from layer5_llm_summarizer import generate_campaign_action_plan
from business_config import CONFIG
from pipeline_logger import log
from crm_export import export_customer_segments_safe, export_segments_safe

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
    # so they match layer4's _normalize_to_pos_format output.
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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_rfm[["recency", "frequency", "monetary"]])

    best_k = 3
    best_score = -1
    max_k = min(5, len(df_rfm) - 1)

    for k in range(2, max_k + 1):
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

    # Rank-based assignment — score each cluster to ensure unique names
    # RFM score: high monetary + high frequency + low recency = best
    max_r = cluster_stats["recency"].max() or 1
    max_f = cluster_stats["frequency"].max() or 1
    max_m = cluster_stats["monetary"].max() or 1
    cluster_stats["rfm_score"] = (
        (1 - cluster_stats["recency"] / max_r) * 0.3 +
        (cluster_stats["frequency"] / max_f) * 0.3 +
        (cluster_stats["monetary"] / max_m) * 0.4
    )
    cluster_stats = cluster_stats.sort_values("rfm_score", ascending=False).reset_index(drop=True)

    name_pool = [
        "1. Champions & Loyal Customers",
        "2. Potential Loyalists",
        "3. At-Risk / Churning Customers",
        "4. Promising & Recent Customers",
        "5. Hibernating Customers",
    ]

    segments = []
    for rank, (_, row) in enumerate(cluster_stats.iterrows()):
        seg_name = name_pool[rank] if rank < len(name_pool) else f"{rank+1}. Segment {rank+1}"
        segments.append({
            "cluster_id": int(row["cluster"]),
            "segment_name": seg_name,
            "customer_count": int(row["count"]),
            "avg_recency_days": round(row["recency"], 1),
            "avg_frequency": round(row["frequency"], 1),
            "avg_monetary": round(row["monetary"], 2)
        })

    result = sorted(segments, key=lambda x: x["segment_name"])
    log("L1", "segmentation_complete",
        optimal_k=best_k,
        silhouette_score=round(best_score, 4),
        segments=[s["segment_name"] for s in result],
        unique_customers=int(df_rfm["customer_id"].nunique()))
    return result, df_rfm

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

    co_matrix = basket_bool.T.dot(basket_bool)
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
        seg_basket = basket_bool.loc[basket_bool.index.isin(seg_invoices)]

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

    log("L2", "mba_complete",
        segments_enriched=len(enriched_segments),
        top_global_pair=f"{global_a} + {global_b}",
        top_global_cooccurrence=global_count)
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
                "send_date":           send_date.date(),
                "send_window":         send_window,
                "send_time_hint":      send_time,
                "channel":             "Email/SMS",
                "should_send":         1,
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
# Main Execution
# ==========================================
if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/transactions.csv"
    print("🚀 เริ่มต้นรัน CSV MLOps Campaign Pipeline...\n")
    log("L0", "pipeline_start", csv_file=csv_file)
    _pipeline_start = time.monotonic()
    try:
        from layer4_uplift_margin import run_layer4

        _t0 = time.monotonic()
        df_raw = load_and_preprocess_csv(csv_file)
        log("L0", "layer_complete", duration_ms=round((time.monotonic() - _t0) * 1000), rows_out=len(df_raw))

        _t0 = time.monotonic()
        _rows_in = len(df_raw)
        segments, df_rfm = layer1_csv_segmentation(df_raw)
        log("L1", "layer_complete", duration_ms=round((time.monotonic() - _t0) * 1000),
            rows_in=_rows_in, rows_out=int(df_rfm["customer_id"].nunique()))

        _t0 = time.monotonic()
        _rows_in = int(df_rfm["customer_id"].nunique())
        segments, df_rfm = layer2_csv_mba(df_raw, segments, df_rfm)
        export_customer_segments_safe(df_rfm, segments)  # customer -> segment, for the CRM API
        log("L2", "layer_complete", duration_ms=round((time.monotonic() - _t0) * 1000),
            rows_in=_rows_in, rows_out=len(segments))

        _t0 = time.monotonic()
        validated_payload, product_cogs, default_cogs = layer3_and_4_processing(segments, df_raw, df_rfm)
        log("L3", "layer_complete", duration_ms=round((time.monotonic() - _t0) * 1000),
            rows_in=len(segments), rows_out=len(validated_payload["target_segments"]))

        print("\n🔍 [Layer 4] รัน LightGBM Uplift + CFO Guardrail...")
        # Resolve promo label source to prevent cross-dataset contamination when
        # running from any working directory. Rules (in priority order):
        # 1. POS file has its own treatment columns → use it directly (Format 1)
        # 2. POS filename contains "marketing_campaign" → load companion
        #    marketing_campaign.csv from the same directory (Format 2)
        # 3. Otherwise → None (RFM proxy fallback)
        _pos_abs = os.path.abspath(csv_file)
        _pos_dir = os.path.dirname(_pos_abs)
        _pos_name = os.path.basename(_pos_abs).lower()
        _header = pd.read_csv(_pos_abs, nrows=0).columns.tolist()
        if 'is_treated' in _header and 'post_promo_visit' in _header:
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
            promo_source=_promo_source,
            promo_path=str(promo_path),
            csv_file=csv_file)

        _t0 = time.monotonic()
        layer4_df = run_layer4(pos_file_path=csv_file, promo_response_path=promo_path,
                               product_cogs_dict=product_cogs, default_cogs=default_cogs)
        _l4_rows = len(layer4_df) if layer4_df is not None else 0
        log("L4", "layer_complete", duration_ms=round((time.monotonic() - _t0) * 1000),
            rows_out=_l4_rows,
            broadcast_count=int((layer4_df["should_send"] == 1).sum()) if layer4_df is not None else 0,
            suppressed_count=int((layer4_df["should_send"] == 0).sum()) if layer4_df is not None else 0)

        validated_payload = merge_layer4_into_payload(validated_payload, layer4_df, df_rfm)
        export_segments_safe(validated_payload)  # for the CRM API; never aborts the pipeline

        print("\n🤖 [Layer 5] ส่งข้อมูลให้ Local LLM สร้าง Action Plan...")
        print("\n" + "="*50)
        _t0 = time.monotonic()
        final_plan = generate_campaign_action_plan(validated_payload)
        log("L5", "layer_complete", duration_ms=round((time.monotonic() - _t0) * 1000))

        log("L0", "pipeline_end", status="SUCCESS",
            total_duration_ms=round((time.monotonic() - _pipeline_start) * 1000))
    except FileNotFoundError as exc:
        log("L0", "pipeline_error", level="ERROR", exc=str(exc), csv_file=csv_file)
        print(f"⚠️ ไม่พบไฟล์ {csv_file} กรุณานำไฟล์ CSV วางไว้ในโฟลเดอร์ data/ หรือปรับ path ให้ถูกต้องครับ")
    except Exception as exc:
        log("L0", "pipeline_error", level="ERROR", exc=type(exc).__name__ + ": " + str(exc))
        raise
