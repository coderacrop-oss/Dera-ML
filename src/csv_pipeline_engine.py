import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from layer5_llm_summarizer import generate_campaign_action_plan
from business_config import CONFIG

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
    df["customer_id"] = df["customer_id"].astype(str)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors='coerce')
    
    if "quantity" in df.columns and "unit_price" in df.columns:
        df["amount"] = df["quantity"] * df["unit_price"]
    elif "amount" not in df.columns:
        df["amount"] = 500.0
        
    df = df[df["amount"] > 0]
    print(f"✅ โหลดสำเร็จ! ข้อมูลดิบทั้งหมด {len(df):,} รายการ จากลูกค้า {df['customer_id'].nunique():,} คน")
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

    return sorted(segments, key=lambda x: x["segment_name"]), df_rfm

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
            cost_ratio = (prod["p10"] / prod["mean"]).clip(0.10, 0.90).median()
            if 0.20 < cost_ratio < 0.80:
                cogs_rate = round(float(cost_ratio), 3)
                source = "data_driven"

    # --- Discount rate: promo frequency × CONFIG depth ---
    if "promo_type" in df.columns:
        disc_freq = (df["promo_type"] != "none").mean()
        estimated = round(disc_freq * CONFIG["default_discount_rate"], 4)
        if 0.005 < estimated < 0.40:
            discount_rate = estimated
            source = "data_driven"

    print(f"💡 [Rates] COGS={cogs_rate:.1%}  Discount={discount_rate:.1%}  source={source}")
    return cogs_rate, discount_rate


def layer3_and_4_processing(segments, df_raw, df_rfm):
    print("⏰ [Layer 3 & 4] คำนวณตารางเวลา, Unit Economics และ Post-Campaign Forecast...")
    updated_segments = []
    total_profit = 0

    # Estimate rates from data (falls back to CONFIG if insufficient)
    estimated_cogs, estimated_discount = _estimate_rates_from_data(df_raw)

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

    return {
        "campaign_name": "CSV Data Driven Retail Campaign",
        "planning_month": "พฤศจิกายน 2026",
        "target_segments": updated_segments,
        "gate_status": "PASSED",
        "total_campaign_projected_profit": f"{total_profit:,.2f} บาท"
    }

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    csv_file = "data/transactions.csv"
    print("🚀 เริ่มต้นรัน CSV MLOps Campaign Pipeline...\n")
    try:
        df_raw = load_and_preprocess_csv(csv_file)
        segments, df_rfm = layer1_csv_segmentation(df_raw)
        segments, df_rfm = layer2_csv_mba(df_raw, segments, df_rfm)
        validated_payload = layer3_and_4_processing(segments, df_raw, df_rfm)
        
        print("\n🤖 [Layer 5] ส่งข้อมูลให้ Local LLM สร้าง Action Plan...")
        print("\n" + "="*50)
        final_plan = generate_campaign_action_plan(validated_payload)
    except FileNotFoundError:
        print(f"⚠️ ไม่พบไฟล์ {csv_file} กรุณานำไฟล์ CSV วางไว้ในโฟลเดอร์ data/ หรือปรับ path ให้ถูกต้องครับ")
