import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn.calibration import CalibratedClassifierCV
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from business_config import CONFIG, REGISTRY_PATH
from peak_hours import write_peak_hours
from atomic_io import atomic_write_text
from model_registry import ModelRegistry, RegistryError
from pipeline_logger import log


# ─────────────────────────────────────────────────────────────────────────────
# Uplift Meta-learners
# ─────────────────────────────────────────────────────────────────────────────

class XLearnerModel:
    """X-learner meta-learner (Künzel et al. 2019) for heterogeneous treatment effects.

    Handles imbalanced treatment/control groups better than T-learner by imputing
    counterfactual outcomes and learning CATE in two stages weighted by propensity.
    """
    def __init__(self, mu0, mu1, tau0, tau1, propensity):
        self.mu0 = mu0              # P(Y=1 | T=0, X) — base learner on control
        self.mu1 = mu1              # P(Y=1 | T=1, X) — base learner on treated
        self.tau0 = tau0            # E[τ | X] estimated from control (regressor)
        self.tau1 = tau1            # E[τ | X] estimated from treated (regressor)
        self.propensity = propensity  # P(T=1 | X)

    def predict_uplift(self, X):
        """CATE estimate: e(x)·τ₀(x) + (1-e(x))·τ₁(x)"""
        e = np.clip(self.propensity.predict_proba(X)[:, 1], 0.05, 0.95)
        return e * self.tau0.predict(X) + (1 - e) * self.tau1.predict(X)

    def predict_control_proba(self, X):
        return self.mu0.predict_proba(X)[:, 1]

    @property
    def n_features_(self):
        # CalibratedClassifierCV uses n_features_in_; raw LGBMClassifier uses n_features_
        return getattr(self.mu0, 'n_features_in_', getattr(self.mu0, 'n_features_', None))


class TLearnerModel:
    """Backward-compatible wrapper for legacy T-learner champion models."""
    def __init__(self, model_t, model_c):
        self.mu1 = model_t
        self.mu0 = model_c

    def predict_uplift(self, X):
        return self.mu1.predict_proba(X)[:, 1] - self.mu0.predict_proba(X)[:, 1]

    def predict_control_proba(self, X):
        return self.mu0.predict_proba(X)[:, 1]

    @property
    def n_features_(self):
        # CalibratedClassifierCV uses n_features_in_; raw LGBMClassifier uses n_features_
        return getattr(self.mu0, 'n_features_in_', getattr(self.mu0, 'n_features_', None))


def _with_names(model, features):
    """Remember the exact feature list/order the model was fitted on (serving verifies it)."""
    model.feature_names_ = list(features)
    return model


def _train_xlearner(treat_data, ctrl_data, features, n_estimators=200):
    """Train X-learner meta-learner (5 LightGBM models total)."""
    n_treat = len(treat_data)
    n_ctrl  = len(ctrl_data)
    print(f"🔬 [X-learner] Training — treated={n_treat:,}, control={n_ctrl:,}, "
          f"features={len(features)}, n_estimators={n_estimators}")

    # Fallback to T-learner if either group is too small for X-learner stages
    if n_treat < 50 or n_ctrl < 50:
        print("⚠️  [X-learner] กลุ่มเล็กเกินไป — ใช้ T-learner แทน")
        mu1 = lgb.LGBMClassifier(n_estimators=n_estimators, random_state=42,
                                  reg_lambda=1.0, min_child_samples=20, subsample=0.8,
                                  colsample_bytree=0.8, verbose=-1)
        mu0 = lgb.LGBMClassifier(n_estimators=n_estimators, random_state=42,
                                  reg_lambda=1.0, min_child_samples=20, subsample=0.8,
                                  colsample_bytree=0.8, verbose=-1)
        mu1.fit(treat_data[features], treat_data['converted'])
        mu0.fit(ctrl_data[features],  ctrl_data['converted'])
        return _with_names(TLearnerModel(mu1, mu0), features)

    # Stage 1: Base learners (outcome models) — calibrated for reliable P(Y=1|X)
    # Raw LightGBM probabilities are uncalibrated: predict_proba(X) = 0.7 ≠ 70% purchase rate.
    # Miscalibrated P(Y=1) propagates directly to profit estimates via:
    #   baseline_profit = gross_margin * p_baseline
    #   incremental     = gross_margin * (p_treated - p_baseline) - promo_cost
    # CalibratedClassifierCV(isotonic, cv=3) fits an isotonic regression on OOF predictions
    # to map raw scores → calibrated probabilities. Also used in Stage 2 cross-predictions.
    mu1 = CalibratedClassifierCV(
        lgb.LGBMClassifier(n_estimators=n_estimators, random_state=42,
                           reg_lambda=1.0, min_child_samples=20, subsample=0.8,
                           colsample_bytree=0.8, verbose=-1),
        method='isotonic', cv=5
    )
    mu0 = CalibratedClassifierCV(
        lgb.LGBMClassifier(n_estimators=n_estimators, random_state=42,
                           reg_lambda=1.0, min_child_samples=20, subsample=0.8,
                           colsample_bytree=0.8, verbose=-1),
        method='isotonic', cv=5
    )
    mu1.fit(treat_data[features], treat_data['converted'])
    mu0.fit(ctrl_data[features],  ctrl_data['converted'])

    # Stage 2: Imputed treatment effects
    # D1 = Y_i − μ₀(X_i) for treated  (actual − predicted counterfactual)
    # D0 = μ₁(X_i) − Y_i for control  (predicted treated − actual)
    D1 = (treat_data['converted'].values.astype(float)
          - mu0.predict_proba(treat_data[features])[:, 1])
    D0 = (mu1.predict_proba(ctrl_data[features])[:, 1]
          - ctrl_data['converted'].values.astype(float))

    # Stage 3: CATE estimators (regressors on imputed effects)
    tau1 = lgb.LGBMRegressor(n_estimators=n_estimators, random_state=42, max_depth=4,
                              reg_lambda=1.0, min_child_samples=20, subsample=0.8,
                              colsample_bytree=0.8, verbose=-1)
    tau0 = lgb.LGBMRegressor(n_estimators=n_estimators, random_state=42, max_depth=4,
                              reg_lambda=1.0, min_child_samples=20, subsample=0.8,
                              colsample_bytree=0.8, verbose=-1)
    tau1.fit(treat_data[features], D1)
    tau0.fit(ctrl_data[features],  D0)

    # Propensity model: P(T=1 | X) — weights τ₀ vs τ₁ by treatment probability
    all_data   = pd.concat([treat_data, ctrl_data], ignore_index=True)
    _prop_base = lgb.LGBMClassifier(n_estimators=n_estimators, random_state=42,
                                    reg_lambda=1.0, min_child_samples=20, subsample=0.8,
                                    colsample_bytree=0.8, verbose=-1)
    propensity = CalibratedClassifierCV(_prop_base, method='isotonic', cv=3)
    propensity.fit(all_data[features], all_data['was_treated'])

    print("✅ [X-learner] Training complete (mu0, mu1, tau0, tau1, propensity)")
    model = _with_names(XLearnerModel(mu0, mu1, tau0, tau1, propensity), features)
    # feature_baseline_stats is set by train_champion after training,
    # using customer-level cust_agg so it matches the serving distribution.
    model.feature_baseline_stats = {}
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Internal Layer 1: RFM + K-Means
# ─────────────────────────────────────────────────────────────────────────────
def run_layer1(pos_file_path="data/raw_pos_transactions.csv"):
    if not os.path.exists(pos_file_path):
        print(f"❌ ไม่พบไฟล์ข้อมูล: {pos_file_path}")
        return None
    df = pd.read_csv(pos_file_path)
    initial_rows = len(df)
    df = df[df['total_price'] > 0].copy()
    Q3 = df['total_price'].quantile(0.75)
    IQR = Q3 - df['total_price'].quantile(0.25)
    df_clean = df[df['total_price'] <= Q3 + 3.0 * IQR].copy()
    filtered = initial_rows - len(df_clean)
    if filtered > 0:
        print(f"🧹 [Data Hygiene]: คัดกรองข้อมูลขยะและ Outliers ออกไป {filtered} รายการ")
    now = pd.to_datetime(df_clean['timestamp']).max() + pd.Timedelta(days=1)
    rfm = df_clean.groupby('customer_id').agg(
        recency=('timestamp', lambda x: (now - pd.to_datetime(x).max()).days),
        frequency=('transaction_id', 'count'),
        monetary=('total_price', 'sum')
    ).reset_index()
    rfm['rfm_decile_segment'] = pd.qcut(
        rfm['monetary'].rank(method='first'), 10, labels=False
    ).astype(int) + 1
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])
    best_k, best_score = 2, -1
    for k in range(2, min(6, len(rfm))):
        score = silhouette_score(X, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X))
        if score > best_score:
            best_score, best_k = score, k
    rfm['kmeans_cluster'] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)
    print(f"✅ [Layer 1 Valuation] K-Means cluster จริง: k={best_k} (Silhouette={best_score:.3f})")
    os.makedirs("data/processed", exist_ok=True)
    rfm.to_csv("data/processed/layer1_dual_valuation.csv", index=False)
    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# Internal Layer 2: Market Basket Analysis
# ─────────────────────────────────────────────────────────────────────────────
def run_layer2(pos_file_path="data/raw_pos_transactions.csv"):
    print("=== กำลังประมวลผล Layer 2: Preference & Bundling (Market Basket Analysis) ===")
    df = pd.read_csv(pos_file_path)
    top_products = df['product_name'].value_counts().head(150).index
    df = df[df['product_name'].isin(top_products)]
    n_inv = df['transaction_id'].nunique()
    min_sup = 0.05 if n_inv > 5000 else (0.02 if n_inv > 1000 else 0.01)
    basket = pd.crosstab(df['transaction_id'], df['product_name']).map(lambda x: 1 if x > 0 else 0)
    freq = apriori(basket, min_support=min_sup, use_colnames=True)
    if freq.empty:
        print("คำเตือน: ไม่พบ Association Rules ที่ตรงตามเงื่อนไข")
        return None
    rules = association_rules(freq, metric="lift", min_threshold=1.0)
    if rules.empty:
        print("คำเตือน: ไม่พบ Association Rules ที่ตรงตามเงื่อนไข")
        return None
    rules = rules.sort_values('lift', ascending=False)
    top = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(3).copy()
    top['antecedents'] = top['antecedents'].apply(lambda x: list(x)[0] if len(x) == 1 else str(x))
    top['consequents'] = top['consequents'].apply(lambda x: list(x)[0] if len(x) == 1 else str(x))
    os.makedirs("data/processed", exist_ok=True)
    top.to_csv("data/processed/layer2_top_bundles.csv", index=False)
    print(f"บันทึกคู่สินค้า Cross-Selling (Top 3 Bundles) สำเร็จ: data/processed/layer2_top_bundles.csv")
    print("\n--- ตัวอย่าง Top 3 Product Bundles ---")
    print(top)
    return top


# ─────────────────────────────────────────────────────────────────────────────
# Internal Layer 3: Context Detection (Peak/Off-peak + Inventory)
# ─────────────────────────────────────────────────────────────────────────────
def run_layer3(pos_file_path="data/raw_pos_transactions.csv"):
    print("=== กำลังประมวลผล Layer 3: Dynamic Context (Peak/Off-Peak & Inventory Check) ===")
    df = pd.read_csv(pos_file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    bad = df['timestamp'].isna().sum()
    if bad > 0:
        print(f"⚠️ [Layer 3] timestamp แปลงไม่ได้ {bad} แถว — ข้ามแถวเหล่านั้น")
        df = df.dropna(subset=['timestamp'])
    biz_tz = CONFIG["business_tz"]
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(biz_tz, ambiguous='infer', nonexistent='shift_forward')
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert(biz_tz)
    # Data-driven peak detection: hours with transaction volume > mean + 0.5*std
    hourly_counts = df['timestamp'].dt.hour.value_counts()
    mean_vol = hourly_counts.mean()
    std_vol  = hourly_counts.std()
    peak_threshold = mean_vol + 0.5 * std_vol
    peak_hours_data = set(hourly_counts[hourly_counts > peak_threshold].index.tolist())
    data_driven = len(peak_hours_data) > 0

    # Fallback to config if data has no spread (e.g. all same hour)
    if len(peak_hours_data) == 0:
        peak_hours_data = set(range(CONFIG["peak_hour_start"], CONFIG["peak_hour_end"] + 1))
        print(f"ℹ️  [Layer 3] Peak hours: config fallback {sorted(peak_hours_data)}")
    else:
        print(f"ℹ️  [Layer 3] Peak hours (data-driven, threshold={peak_threshold:.0f} txns): {sorted(peak_hours_data)}")
    # Persist the result so downstream steps (campaign_scheduler, CRM API) reuse it instead of recomputing.
    write_peak_hours(peak_hours_data, set(hourly_counts.index.tolist()) - peak_hours_data, biz_tz,
                     peak_threshold if data_driven else None, "data" if data_driven else "config_fallback")

    df['time_context'] = df['timestamp'].dt.hour.apply(
        lambda h: 'Peak' if h in peak_hours_data else 'Off-Peak'
    )
    restock = CONFIG.get("stock_restock_cycle_days", 14)
    n_days = max(1, (df['timestamp'].max() - df['timestamp'].min()).days)
    total_qty = df.groupby('product_name')['quantity'].sum()
    daily_rate = total_qty / n_days
    initial_est = (daily_rate * restock * CONFIG["stock_buffer_multiplier"]).clip(lower=10).round().astype(int)
    recent_cutoff = df['timestamp'].max() - pd.Timedelta(days=restock)
    recent_qty = (
        df[df['timestamp'] >= recent_cutoff].groupby('product_name')['quantity'].sum()
        .reindex(total_qty.index, fill_value=0)
    )
    stock = total_qty.reset_index().rename(columns={'quantity': 'total_quantity'})
    stock['initial_stock_est'] = initial_est.values
    stock['recent_demand'] = recent_qty.values
    stock['remaining_stock'] = (stock['initial_stock_est'] - stock['recent_demand']).clip(lower=0)
    stock['in_stock'] = stock['remaining_stock'] > (stock['initial_stock_est'] * 0.20).clip(lower=CONFIG["low_stock_threshold"])
    stock = stock.rename(columns={'total_quantity': 'quantity'})
    context_df = df[['transaction_id', 'customer_id', 'product_name', 'timestamp', 'time_context']].drop_duplicates()
    context_df = context_df.merge(stock[['product_name', 'remaining_stock', 'in_stock']], on='product_name', how='left')
    os.makedirs("data/processed", exist_ok=True)
    context_df.to_csv("data/processed/layer3_context_flags.csv", index=False)
    return context_df

def _normalize_to_pos_format(file_path: str) -> str:
    """
    Detect CSV format and normalize to raw_pos schema if needed.
    Returns the original path if already normalized, otherwise writes to
    data/processed/normalized_for_layer4.csv and returns that path.
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8', nrows=1)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1', nrows=1)

    # Already in pos format
    if {'customer_id', 'timestamp', 'total_price', 'transaction_id'}.issubset(df.columns):
        return file_path

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')

    col_map = {
        'CustomerID':  'customer_id',
        'InvoiceDate': 'timestamp',
        'Description': 'product_name',
        'Quantity':    'quantity',
        'UnitPrice':   'unit_price',
        'InvoiceNo':   'invoice_id',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Accept 'transaction_date' as alias for 'timestamp'
    if 'timestamp' not in df.columns and 'transaction_date' in df.columns:
        df = df.rename(columns={'transaction_date': 'timestamp'})

    df = df.dropna(subset=['customer_id'])
    # CustomerID may be float (e.g. 17850.0) — convert to clean int string
    try:
        df['customer_id'] = df['customer_id'].astype(float).astype(int).astype(str)
    except (ValueError, TypeError):
        df['customer_id'] = df['customer_id'].astype(str).str.strip()
    df = df[df['customer_id'].str.strip() != '']
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['customer_id', 'timestamp'])

    if 'total_price' not in df.columns:
        df['total_price'] = df['quantity'] * df['unit_price']
    df = df[df['total_price'] > 0]

    # Use invoice_id as basket unit (customer + date proxy if no InvoiceNo)
    if 'invoice_id' not in df.columns:
        df['invoice_id'] = df['customer_id'] + '_' + df['timestamp'].dt.date.astype(str)

    # transaction_id = invoice_id so layer2_bundling groups correctly into baskets
    if 'transaction_id' not in df.columns:
        df['transaction_id'] = df['invoice_id']

    out_path = 'data/processed/normalized_for_layer4.csv'
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"📄 [Layer 4] Normalized {len(df):,} rows → {out_path}")
    return out_path


def _load_real_labels(promo_path: str) -> pd.DataFrame | None:
    """
    Load real promotion response labels. Supports two formats:
      1. marketing_campaign.csv  — AcceptedCmp1-5, Response, Income
      2. raw_pos_transactions.csv — is_treated, post_promo_visit, promo_type
    Returns DataFrame with [customer_id, was_treated, converted, income,
    total_spend, times_accepted] or None if file is invalid.
    """
    if promo_path is None or not os.path.exists(promo_path):
        return None
    try:
        df = pd.read_csv(promo_path, sep=';')
        if len(df.columns) == 1:          # sep ผิด → ลอง comma
            df = pd.read_csv(promo_path)
    except Exception:
        try:
            df = pd.read_csv(promo_path)
        except Exception:
            return None

    # --- Format 1: raw_pos (transaction-level with is_treated + post_promo_visit) ---
    if 'is_treated' in df.columns and 'post_promo_visit' in df.columns:
        # Normalize customer_id column name (handle CustomerID, customer_id variants)
        cid_map = {c: 'customer_id' for c in df.columns if c.lower() in ('customerid', 'customer_id')}
        df = df.rename(columns=cid_map)
        if 'customer_id' not in df.columns:
            return None
        # Normalize timestamp column
        ts_map = {c: 'timestamp' for c in df.columns if c.lower() in ('invoicedate', 'timestamp', 'transaction_date')}
        df = df.rename(columns=ts_map)
        # Normalize total_price column
        tp_map = {c: 'total_price' for c in df.columns
                  if c.lower() in ('unitprice', 'total_price', 'amount') and 'total_price' not in df.columns}
        df = df.rename(columns=tp_map)
        if 'total_price' not in df.columns and 'quantity' in df.columns and 'UnitPrice' in df.columns:
            df['total_price'] = df['quantity'] * df['UnitPrice']
        # Normalize transaction_id
        if 'transaction_id' not in df.columns:
            df['transaction_id'] = df.index.astype(str)
        try:
            df['customer_id'] = df['customer_id'].astype(float).astype(int).astype(str)
        except (ValueError, TypeError):
            df['customer_id'] = df['customer_id'].astype(str).str.strip()
        # Optional columns — fill if missing
        if 'promo_applied' not in df.columns:
            df['promo_applied'] = df['is_treated']
        if 'transaction_id' not in df.columns:
            df['transaction_id'] = df.index.astype(str)
        if 'total_price' not in df.columns and 'total_price' not in df.columns:
            df['total_price'] = 0.0
        cust = df.groupby('customer_id').agg(
            was_treated      =('is_treated',      'max'),   # เคยได้รับโปรอย่างน้อย 1 ครั้ง
            converted        =('post_promo_visit', 'mean'), # สัดส่วน visits ที่กลับมาหลังโปร (0-1)
            times_accepted   =('promo_applied',    'sum'),  # จำนวนครั้งที่ใช้โปรจริง
            total_spend      =('total_price',      'sum'),
            n_transactions   =('transaction_id',   'count'),
        ).reset_index()
        # แปลง conversion rate → binary. ถ้าเป็น 0/1 อยู่แล้ว (label ต่อลูกค้า) ใช้ตามนั้น: กฎ "> median" จะลบ label ทิ้งทั้งหมด
        # เมื่อคนกลับมาซื้อเกินครึ่ง (median = 1 ไม่มีใครสูงกว่า 1) ส่วนอัตราที่เป็นเศษส่วน: > median rate = converted
        rates = cust['converted']
        if rates.dropna().isin([0, 1]).all():  # NaN = ลูกค้าที่ไม่มีค่าเลย ไม่ควรทำให้ทั้งไฟล์กลับไปใช้กฎ median
            cust['converted'] = rates.fillna(0).astype(int)
        else:
            median_conv = rates.median()  # full population, not treated-only
            cust['converted'] = (rates > median_conv).astype(int)
        cust['income'] = 0.0   # ไม่มี income ใน raw_pos

        # เพิ่ม promo_type dummies เป็น features เพิ่มเติม
        if 'promo_type' in df.columns:
            promo_dummies = pd.get_dummies(df['promo_type'], prefix='promo').groupby(
                df['customer_id'].values).max().reset_index()
            promo_dummies.columns = ['customer_id'] + list(promo_dummies.columns[1:])
            cust = cust.merge(promo_dummies, on='customer_id', how='left').fillna(0)

        result = cust
        source = "raw_pos (is_treated + post_promo_visit)"

    # --- Format 2: marketing_campaign (customer-level AcceptedCmp + Response) ---
    elif 'Response' in df.columns and 'ID' in df.columns:
        cmp_cols = [c for c in df.columns if c.startswith('AcceptedCmp')]
        if not cmp_cols:
            print("⚠️ [Layer 4] ไม่พบ AcceptedCmp columns — ข้าม")
            return None
        spend_cols = [c for c in ['MntWines','MntFruits','MntMeatProducts',
                                   'MntFishProducts','MntSweetProducts','MntGoldProds']
                      if c in df.columns]
        df['customer_id']    = df['ID'].astype(int).astype(str)
        df['was_treated']    = df[cmp_cols].max(axis=1).astype(int)
        df['converted']      = df['Response'].astype(int)
        df['times_accepted'] = df[cmp_cols].sum(axis=1)
        df['total_spend']    = df[spend_cols].sum(axis=1) if spend_cols else 0.0
        df['income']         = pd.to_numeric(df.get('Income', 0), errors='coerce').fillna(0)
        result = df[['customer_id', 'was_treated', 'converted',
                     'income', 'total_spend', 'times_accepted']].copy()
        source = "marketing_campaign (AcceptedCmp + Response)"

    else:
        print("⚠️ [Layer 4] ไม่รู้จัก format ของ promo file — ข้าม")
        return None

    n_treated   = int(result['was_treated'].sum())
    n_converted = int(result['converted'].sum())
    print(f"✅ [Layer 4] Real labels loaded [{source}]: {len(result):,} customers | "
          f"treated={n_treated} ({n_treated/len(result)*100:.1f}%) | "
          f"converted={n_converted} ({n_converted/len(result)*100:.1f}%)")
    return result


@dataclass
class Layer4Context:
    """Everything Layer 4 derives from one shop's CSV before any model is fitted or applied."""
    raw_df: pd.DataFrame
    cust_agg: pd.DataFrame
    features: list
    rfm_df: pd.DataFrame
    context_df: pd.DataFrame
    top_bundle_name: str
    product_cogs: dict
    default_cogs: float
    label_mode: str  # "real" (promotion-response labels) | "proxy" (temporal-split + random treatment)


def build_layer4_context(pos_file_path, promo_response_path=None, product_cogs_dict=None, default_cogs=None):
    if not os.path.exists(pos_file_path):
        raise FileNotFoundError(pos_file_path)

    norm_path = _normalize_to_pos_format(pos_file_path)
    raw_df = pd.read_csv(norm_path)
    raw_df['customer_id'] = raw_df['customer_id'].astype(str)

    # 1. ตรวจจับ Spike และสภาวะตลาดเบื้องต้น
    mean_price = raw_df['total_price'].mean()
    std_price = raw_df['total_price'].std()
    threshold_spike = mean_price + (2.5 * std_price)
    raw_df['is_campaign_spike'] = raw_df['total_price'].apply(lambda x: 1 if x > threshold_spike else 0)

    rfm_df = run_layer1(norm_path)
    rfm_df['customer_id'] = rfm_df['customer_id'].astype(str)
    bundles_df = run_layer2(norm_path)
    context_df = run_layer3(norm_path)
    context_df['customer_id'] = context_df['customer_id'].astype(str)

    top_bundle_name = "เซตคู่คุ้มค่า (Bundle)"
    if bundles_df is not None and not bundles_df.empty:
        b1 = bundles_df.iloc[0]
        top_bundle_name = f"Bundle: {b1['antecedents']} + {b1['consequents']}"

    # --- Labels: real promotion response ถ้ามี, fallback เป็น RFM proxy ---
    rfm_labels = rfm_df[['customer_id', 'rfm_decile_segment', 'frequency']].copy()
    median_decile = rfm_labels['rfm_decile_segment'].median()
    raw_df = raw_df.merge(rfm_labels, on='customer_id', how='left')
    raw_df['rfm_decile_segment'] = raw_df['rfm_decile_segment'].fillna(median_decile)
    raw_df['frequency'] = raw_df['frequency'].fillna(1)

    real_labels = _load_real_labels(promo_response_path)
    use_real_labels = real_labels is not None
    promo_dummy_cols = []

    if use_real_labels:
        print("🎯 [Layer 4] ใช้ Real Promotion Response labels แทน RFM proxy")
        raw_df['customer_id'] = raw_df['customer_id'].astype(str)
        real_labels['customer_id'] = real_labels['customer_id'].astype(str)
        raw_df = raw_df.merge(
            real_labels[['customer_id', 'was_treated', 'converted',
                         'income', 'total_spend', 'times_accepted']],
            on='customer_id', how='left'
        )
        # fallback สำหรับลูกค้าที่ไม่มีใน real_labels:
        # ใช้ temporal split แทน RFM proxy (ป้องกัน leakage)
        _ts = pd.to_datetime(raw_df['timestamp'], errors='coerce')
        _split = _ts.quantile(0.75)
        _post_custs = set(raw_df.loc[_ts > _split, 'customer_id'].unique())
        raw_df['_temporal_conv'] = raw_df['customer_id'].isin(_post_custs).astype(int)
        # ลูกค้าที่ไม่อยู่ใน real_labels → ถือเป็น control (ไม่มีข้อมูลว่าถูก treat)
        raw_df['was_treated'] = raw_df['was_treated'].fillna(0).astype(int)
        raw_df['converted']   = raw_df['converted'].fillna(raw_df['_temporal_conv']).astype(int)
        raw_df = raw_df.drop(columns=['_temporal_conv'])
        raw_df['income']         = raw_df['income'].fillna(0)
        raw_df['total_spend']    = raw_df['total_spend'].fillna(0)
        raw_df['times_accepted'] = raw_df['times_accepted'].fillna(0)
        for col in promo_dummy_cols:
            if col in raw_df.columns:
                raw_df[col] = raw_df[col].fillna(0)
        promo_dummy_cols = [c for c in real_labels.columns
                            if c.startswith('promo_') and c != 'promo_none']
        # merge promo dummy cols จาก real_labels ลง raw_df
        if promo_dummy_cols:
            promo_merge = real_labels[['customer_id'] + promo_dummy_cols].copy()
            promo_merge['customer_id'] = promo_merge['customer_id'].astype(str)
            raw_df = raw_df.merge(promo_merge, on='customer_id', how='left')
            for col in promo_dummy_cols:
                raw_df[col] = raw_df[col].fillna(0)
        features = ['total_price', 'quantity', 'is_campaign_spike',
                    'rfm_decile_segment', 'frequency',
                    'income', 'total_spend', 'times_accepted'] + promo_dummy_cols
    else:
        # Temporal-split proxy: แทน RFM proxy เพื่อป้องกัน treatment leakage
        # ────────────────────────────────────────────────────────────────────
        # ปัญหาเดิม: was_treated = rfm_decile >= median และ converted = frequency > 1
        # แล้วนำ rfm_decile_segment + frequency ไปเป็น feature ด้วย → โมเดล learn RFM rank
        # ไม่ใช่ causal uplift จริง
        #
        # แนวทางใหม่:
        # 1. converted = customer กลับมาซื้อในช่วง 25% สุดท้ายของ observation window
        #    (temporal retention signal — ไม่เกี่ยวกับ RFM rank โดยตรง)
        # 2. was_treated = random assignment seed=42 (ไม่มีข้อมูล treatment จริง →
        #    random ทำให้ AUUC → 0 ซึ่งเป็นคำตอบที่ถูกต้องและ honest)
        # 3. features: basket signals เท่านั้น (ไม่รวม rfm_decile_segment / frequency)
        # ────────────────────────────────────────────────────────────────────
        print("ℹ️  [Layer 4] Temporal-split proxy (ไม่พบ real promo labels) — ป้องกัน RFM leakage")

        _ts = pd.to_datetime(raw_df['timestamp'], errors='coerce')
        _split_date = _ts.quantile(0.75)

        _pre  = raw_df[_ts <= _split_date].groupby('customer_id')['total_price'].count().rename('n_pre')
        _post = raw_df[_ts >  _split_date].groupby('customer_id')['total_price'].count().rename('n_post')
        _retention = _pre.to_frame().join(_post, how='outer').fillna(0).reset_index()
        _retention['converted'] = (_retention['n_post'] > 0).astype(int)

        # Random treatment: preserved rate ~40%, seeded สำหรับ reproducibility
        _rng = np.random.default_rng(42)
        _retention['was_treated'] = _rng.binomial(1, 0.4, len(_retention))

        raw_df = raw_df.merge(
            _retention[['customer_id', 'converted', 'was_treated']],
            on='customer_id', how='left'
        )
        raw_df['was_treated'] = raw_df['was_treated'].fillna(0).astype(int)
        raw_df['converted']   = raw_df['converted'].fillna(0).astype(int)

        _n_cust  = raw_df['customer_id'].nunique()
        _n_conv  = int(_retention['converted'].sum())
        _n_treat = int(_retention['was_treated'].sum())
        print(f"ℹ️  [Layer 4] split={_split_date.date()} | "
              f"customers={_n_cust} | converted={_n_conv} ({_n_conv/_n_cust*100:.1f}%) | "
              f"treated={_n_treat} ({_n_treat/_n_cust*100:.1f}%)")
        print("⚠️  [Layer 4] Proxy mode: uplift scores = pseudo-CATE "
              "(random treatment → AUUC ≈ 0 คือคำตอบที่ถูกต้อง)")

        # ไม่รวม rfm_decile_segment / frequency ใน features —
        # ตัวแปรเหล่านี้ยังคงอยู่ใน cust_agg เพื่อใช้ใน action rules แต่ไม่ใช่ ML input
        features = ['total_price', 'quantity', 'is_campaign_spike']

    # --- Customer-level aggregation ---
    agg_dict = {
        'total_price':        ['mean', 'sum'],
        'quantity':           'mean',
        'is_campaign_spike':  'mean',
        'rfm_decile_segment': 'first',
        'frequency':          'first',
        'was_treated':        'max',
        'converted':          'max',
    }
    if use_real_labels:
        agg_dict['income']         = 'first'
        agg_dict['total_spend']    = 'first'
        agg_dict['times_accepted'] = 'first'
        for col in promo_dummy_cols:
            if col in raw_df.columns:
                agg_dict[col] = 'first'

    cust_agg = raw_df.groupby('customer_id').agg(agg_dict).reset_index()

    base_cols = ['customer_id', 'total_price', 'total_spent', 'quantity',
                 'is_campaign_spike', 'rfm_decile_segment', 'frequency',
                 'was_treated', 'converted']
    if use_real_labels:
        extra_cols = ['income', 'total_spend', 'times_accepted'] + \
                     [c for c in promo_dummy_cols if c in raw_df.columns]
        cust_agg.columns = base_cols + extra_cols
        # promo dummy cols ที่ไม่มีใน raw_df (มาจาก real_labels เท่านั้น) → fillna 0
        for col in promo_dummy_cols:
            if col not in cust_agg.columns:
                cust_agg[col] = 0
    else:
        cust_agg.columns = base_cols
    cust_agg['avg_spent'] = cust_agg['total_price']

    # ── Per-customer top product (highest cumulative spend) ──────────────────
    # Used to look up product-specific COGS for guardrail calculation
    top_product_map = (
        raw_df.groupby(['customer_id', 'product_name'])['total_price']
        .sum().reset_index()
        .sort_values('total_price', ascending=False)
        .drop_duplicates('customer_id')
        .set_index('customer_id')['product_name']
    )
    cust_agg['top_product'] = cust_agg['customer_id'].map(top_product_map).fillna('')

    # Resolve COGS source: per-product dict > default_cogs param > CONFIG
    _product_cogs = product_cogs_dict or {}
    _default_cogs = default_cogs if default_cogs is not None else CONFIG["cogs_rate"]
    n_matched = sum(1 for p in cust_agg['top_product'] if p in _product_cogs)
    print(f"💡 [Layer 4] Per-product COGS: {n_matched:,}/{len(cust_agg):,} customers matched "
          f"(fallback={_default_cogs:.1%})")

    return Layer4Context(raw_df=raw_df, cust_agg=cust_agg, features=features, rfm_df=rfm_df,
                         context_df=context_df, top_bundle_name=top_bundle_name,
                         product_cogs=_product_cogs, default_cogs=_default_cogs,
                         label_mode="real" if use_real_labels else "proxy")


def decide_for_customers(model, ctx, customer_ids=None):
    """Score customers with `model` and apply the action rules + CFO guardrail.

    Returns (total expected profit, per-customer frame incl. `should_send`)."""
    cust_agg, features = ctx.cust_agg, ctx.features
    rfm_df, context_df = ctx.rfm_df, ctx.context_df
    top_bundle_name = ctx.top_bundle_name
    _product_cogs, _default_cogs = ctx.product_cogs, ctx.default_cogs
    if customer_ids is not None:
        cust_agg = cust_agg[cust_agg['customer_id'].isin(customer_ids)]
    temp_df = cust_agg.copy()
    if len(temp_df) == 0:
        for _c in ('recommended_action', 'expected_profit', 'guardrail_blocked', 'send_window', 'should_send'):
            temp_df[_c] = pd.Series(dtype=object)
        return 0.0, temp_df
    temp_df['uplift_score'] = model.predict_uplift(cust_agg[features]).round(4)
    temp_df['prob_control'] = model.predict_control_proba(cust_agg[features]).round(4)
    # rfm_decile_segment และ frequency มีใน cust_agg แล้ว — merge เฉพาะ columns ที่ยังขาด
    rfm_extra = rfm_df[['customer_id', 'recency', 'monetary', 'kmeans_cluster']].drop_duplicates('customer_id')
    temp_df = temp_df.merge(rfm_extra, on='customer_id', how='left')
    # Assign each customer their MAJORITY time_context (mode across all transactions)
    # so a customer who buys 70% of the time during off-peak hours gets "Off-Peak"
    customer_context = (
        context_df.groupby('customer_id')['time_context']
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else 'Peak')
        .reset_index()
    )
    customer_stock = (
        context_df.groupby('customer_id')['in_stock']
        .agg(lambda x: bool(x.any()))   # in_stock if ANY of their products is in stock
        .reset_index()
    )
    customer_ctx = customer_context.merge(customer_stock, on='customer_id', how='left')
    temp_df = temp_df.merge(customer_ctx, on='customer_id', how='left')

    PROMO_COST_RATES = CONFIG["promo_cost_rates"]

    # Data-adaptive basket thresholds: if dataset avg_spent is far below config
    # threshold (e.g. grocery vs wholesale), scale down to P10/P50 of actual data
    config_low = CONFIG["low_basket_threshold"]
    config_mid = CONFIG["mid_basket_threshold"]
    actual_p10 = float(temp_df['avg_spent'].quantile(0.10))
    actual_p50 = float(temp_df['avg_spent'].quantile(0.50))
    if actual_p50 < config_low:
        low_basket = actual_p10
        mid_basket = actual_p50
        print(f"💡 [Layer 4] Adaptive thresholds: low_basket={low_basket:.2f} (P10), mid_basket={mid_basket:.2f} (P50) [dataset mean below config floor]")
    else:
        low_basket = config_low
        mid_basket = config_mid
    neg_uplift   = CONFIG["negative_uplift_guard"]
    high_uplift  = CONFIG["high_uplift_threshold"]
    vip_rfm      = CONFIG["vip_rfm_threshold"]
    vip_prob     = CONFIG["vip_prob_threshold"]
    spike_rfm    = CONFIG["spike_rfm_threshold"]
    mid_rfm      = CONFIG["mid_rfm_threshold"]

    MIN_MARGIN = CONFIG["min_allowable_net_margin"]

    def calculate_action_and_profit(row):
        uplift       = float(row['uplift_score'])
        prob_c       = float(row['prob_control'])
        rfm_seg      = row['rfm_decile_segment']
        in_stock     = row['in_stock']
        avg_spent    = float(row['avg_spent'])
        is_spike     = float(row['is_campaign_spike'])
        time_context = row.get('time_context', 'Peak')   # "Peak" or "Off-Peak"
        is_off_peak  = (time_context == 'Off-Peak')

        # Per-product COGS: look up customer's top product, fall back to dataset default
        top_prod  = row.get('top_product', '')
        cogs      = _product_cogs.get(top_prod, _default_cogs)
        gross_margin = avg_spent * (1 - cogs)
        p_baseline   = max(0.0, prob_c)
        p_treated    = max(0.0, min(1.0, prob_c + uplift))
        baseline_profit = round(gross_margin * p_baseline, 2)

        # --- Do Nothing cases ---
        if pd.isna(in_stock) or not in_stock:
            return pd.Series(["Do Nothing (Out of Stock)", baseline_profit, 0, time_context])
        if avg_spent < low_basket:
            return pd.Series(["Do Nothing (Low Basket Size)", baseline_profit, 0, time_context])
        if uplift < neg_uplift:
            return pd.Series(["Do Nothing (Negative Uplift Guard)", baseline_profit, 0, time_context])

        # Helper: per-customer CFO Guardrail check
        # Incremental profit = gross_margin * (p_treated - p_baseline) - promo_cost
        # Off-peak allows higher promo cost — justified by driving traffic to a quiet store.
        # Peak uses lower cost — customers are already there, just retain/upsell lightly.
        def _guardrail_ok(promo_cost_rate):
            promo_cost = avg_spent * promo_cost_rate
            gross_rev  = avg_spent * (1 - CONFIG["default_discount_rate"])
            net_profit = gross_rev - avg_spent * cogs - promo_cost
            net_margin = net_profit / gross_rev if gross_rev > 0 else 0
            incremental = gross_margin * (p_treated - p_baseline) - promo_cost
            return net_margin >= MIN_MARGIN and incremental > 0, promo_cost

        # --- Peak-aware send window (for broadcast_list.csv) ---
        # Notify customer 30-60 min BEFORE their typical shopping window opens
        send_window = "Off-Peak Window" if is_off_peak else "Peak Window"

        # --- Active campaign cases with context-aware guardrail ---

        # Festival spike — context-independent, always use festival rate
        if is_spike > 0.5 and rfm_seg >= spike_rfm:
            ok, promo_cost = _guardrail_ok(PROMO_COST_RATES["festival"])
            if not ok:
                return pd.Series(["Do Nothing (CFO Guardrail)", baseline_profit, 1, time_context])
            return pd.Series(["Standard Retention (Festival Guard)",
                              round(gross_margin * p_treated - promo_cost, 2), 0, send_window])

        # Off-Peak customers: goal is to drive traffic when store is quiet
        # Use aggressive traffic-driver discount (off_peak_traffic rate = 0.22)
        # Action name reflects the business intent: pull them in during the quiet window
        if is_off_peak and uplift > 0 and rfm_seg >= mid_rfm:
            ok, promo_cost = _guardrail_ok(PROMO_COST_RATES["off_peak_traffic"])
            if not ok:
                # Too expensive even with off-peak intent → try lighter VIP points
                ok, promo_cost = _guardrail_ok(PROMO_COST_RATES["off_peak_vip"])
                if not ok:
                    return pd.Series(["Do Nothing (CFO Guardrail)", baseline_profit, 1, time_context])
                return pd.Series(["Off-Peak VIP Points: ดึงลูกค้าช่วงเงียบ (แต้มพิเศษ)",
                                  round(gross_margin * p_treated - promo_cost, 2), 0, send_window])
            return pd.Series(["Off-Peak Traffic Driver: ส่วนลดพิเศษช่วงเงียบ",
                              round(gross_margin * p_treated - promo_cost, 2), 0, send_window])

        # Peak customers: already arriving → upsell/reward, lower cost
        if rfm_seg >= vip_rfm and prob_c > vip_prob:
            ok, promo_cost = _guardrail_ok(PROMO_COST_RATES["peak_upsell"] if not is_off_peak
                                           else PROMO_COST_RATES["vip_points"])
            if not ok:
                return pd.Series(["Do Nothing (CFO Guardrail)", baseline_profit, 1, time_context])
            return pd.Series(["VIP Reward: แจกแต้ม X2",
                              round(gross_margin * p_treated - promo_cost, 2), 0, send_window])

        if rfm_seg >= vip_rfm and rfm_seg >= 9:
            ok, promo_cost = _guardrail_ok(PROMO_COST_RATES["peak_upsell"] if not is_off_peak
                                           else PROMO_COST_RATES["vip_points"])
            if not ok:
                return pd.Series(["Do Nothing (CFO Guardrail)", baseline_profit, 1, time_context])
            return pd.Series(["VIP Early Access",
                              round(gross_margin * p_treated - promo_cost, 2), 0, send_window])

        if uplift > high_uplift:
            rate = PROMO_COST_RATES["peak_upsell"] if not is_off_peak else PROMO_COST_RATES["bundle"]
            ok, promo_cost = _guardrail_ok(rate)
            if not ok:
                return pd.Series(["Do Nothing (CFO Guardrail)", baseline_profit, 1, time_context])
            return pd.Series([f"Push Promotion: {top_bundle_name}",
                              round(gross_margin * p_treated - promo_cost, 2), 0, send_window])

        # Default: standard retention, peak uses cheaper rate
        rate = PROMO_COST_RATES["standard"] if not is_off_peak else PROMO_COST_RATES["off_peak"]
        ok, promo_cost = _guardrail_ok(rate)
        if not ok:
            return pd.Series(["Do Nothing (CFO Guardrail)", baseline_profit, 1, time_context])
        return pd.Series(["Standard Retention",
                          round(gross_margin * p_treated - promo_cost, 2), 0, send_window])
        
    res = temp_df.apply(calculate_action_and_profit, axis=1)
    temp_df['recommended_action'] = res[0]
    temp_df['expected_profit']    = res[1]
    temp_df['guardrail_blocked']  = res[2]
    temp_df['send_window']        = res[3]
    temp_df = temp_df.sort_values(by='expected_profit', ascending=False).reset_index(drop=True)
    temp_df['should_send'] = (
        (temp_df['guardrail_blocked'] == 0) &
        (~temp_df['recommended_action'].str.startswith('Do Nothing'))
    ).astype(int)
    return temp_df['expected_profit'].sum(), temp_df


def market_is_recession(raw_df):
    """(is_recession, current_avg_spent): last 25% of the timeline vs the first 75%."""
    ts = pd.to_datetime(raw_df.get('timestamp', pd.NaT), errors='coerce')
    ts_min, ts_max = ts.min(), ts.max()
    if pd.notna(ts_min) and pd.notna(ts_max) and ts_min != ts_max:
        cutoff = ts_min + (ts_max - ts_min) * 0.75
        baseline_avg_spent = raw_df.loc[ts <= cutoff, 'total_price'].mean()
        current_avg_spent = raw_df.loc[ts > cutoff, 'total_price'].mean()
        if pd.isna(baseline_avg_spent) or baseline_avg_spent <= 0:
            baseline_avg_spent = raw_df['total_price'].mean()
        if pd.isna(current_avg_spent) or current_avg_spent <= 0:
            current_avg_spent = raw_df['total_price'].mean()
    else:
        baseline_avg_spent = raw_df['total_price'].mean()
        current_avg_spent = baseline_avg_spent
    return bool(current_avg_spent < (baseline_avg_spent * CONFIG["recession_threshold"])), float(current_avg_spent)


@dataclass
class GateResult:
    passed: bool
    reason: str
    growth_pct: float | None


def gate_decision(challenger_profit, champion_profit, do_nothing_pct, is_recession, current_avg_spent):
    """Advice on whether a challenger beats the champion (same rules as the old in-run gate)."""
    if champion_profit is None:
        return GateResult(True, "บันทึกโมเดลนี้เป็น Champion ตัวแรก", None)
    if champion_profit <= 0:
        return GateResult(False, "กำไรที่ประมาณได้ของ Champion เป็น 0 หรือติดลบ จึงเทียบเป็น % ไม่ได้ — "
                                 "ให้ผู้อนุมัติพิจารณาเอง", None)
    growth_pct = ((challenger_profit - champion_profit) / champion_profit) * 100
    gate_tgt = CONFIG["gate_growth_target_pct"]
    if is_recession:
        dn_thr = CONFIG["do_nothing_recession_threshold"]
        if do_nothing_pct > dn_thr:
            return GateResult(True, f"⚠️ [RECESSION MODE DETECTED]: ตลาดซบเซา (เฉลี่ย {current_avg_spent:,.2f} บาท) -> "
                                    f"อนุมัติโมเดลใหม่ที่เน้น Defensive เพื่อเซฟงบ (Do Nothing = {do_nothing_pct:.1f}%)", growth_pct)
        return GateResult(growth_pct >= gate_tgt, "ตลาดซบเซาแต่โมเดลใหม่ไม่เน้นเซฟงบ", growth_pct)
    if abs(growth_pct) < 0.5:
        return GateResult(False, f"โมเดลใหม่ให้ผลเท่าเดิม ({growth_pct:+.2f}%) — "
                                 f"น่าจะเทรนด้วยข้อมูลชุดเดิม ไม่มี signal ใหม่ คงโมเดลเก่าไว้", growth_pct)
    if growth_pct < 0:
        return GateResult(False, f"โมเดลใหม่ประเมินกำไรได้ต่ำกว่าโมเดลเก่า {growth_pct:+.2f}% — "
                                 f"อาจเกิดจาก dataset คนละชุด (distribution shift) หรือโมเดลเก่า overoptimistic "
                                 f"คงโมเดลเก่าไว้ก่อน แนะนำ: reset champion ถ้าเปลี่ยน dataset ใหม่ทั้งหมด", growth_pct)
    if growth_pct < gate_tgt:
        return GateResult(False, f"โมเดลใหม่ดีขึ้นเล็กน้อย {growth_pct:+.2f}% แต่ยังไม่ถึงเกณฑ์ {gate_tgt:.1f}% "
                                 f"ที่ตั้งไว้ — คงโมเดลเก่าไว้", growth_pct)
    return GateResult(True, f"โมเดลใหม่ดีกว่าโมเดลเก่า {growth_pct:+.2f}% ผ่านเกณฑ์ {gate_tgt:.1f}% — "
                            f"อัปเดต champion", growth_pct)


def _check_features(model, features):
    """The champion must have been fitted on exactly the features this CSV produces."""
    names = getattr(model, "feature_names_", None)
    if names is None:
        # legacy champion (imported as v1) never recorded its feature names: count is all we can verify
        if model.n_features_ != len(features):
            raise RegistryError(f"champion was fitted on {model.n_features_} features but this CSV produces "
                                f"{len(features)} features ({features}); train a new champion for this data")
        log("L4", "legacy_champion_feature_names_unknown", level="WARN", n_features=len(features))
    elif list(names) != list(features):
        raise RegistryError(f"champion expects features {list(names)} but this CSV produces {list(features)}; "
                            f"train a new champion for this data")


def run_layer4(pos_file_path="data/transactions.csv",
               promo_response_path="data/marketing_campaign.csv",
               product_cogs_dict=None,
               default_cogs=None,
               models_root=None):
    """Serving: score every customer with the approved champion. Never trains, never writes the registry."""
    print("=== กำลังประมวลผล Layer 4: Campaign Engine (Serving — ใช้ Champion ที่อนุมัติแล้ว) ===")

    if not os.path.exists(pos_file_path):
        print(f"❌ ไม่พบไฟล์ข้อมูล: {pos_file_path}")
        return

    registry = ModelRegistry(models_root or os.path.dirname(REGISTRY_PATH))
    champion, ref = registry.load_current()   # fails before any output if there is no usable champion

    ctx = build_layer4_context(pos_file_path, promo_response_path, product_cogs_dict, default_cogs)
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

    total_cust = len(detail)
    print(f"\n🎯 การตัดสินใจ Layer 4 ({total_cust:,} ลูกค้า) โดย Champion {ref.version} [{ref.provenance}]:")
    for action, cnt in detail['recommended_action'].value_counts().items():
        icon = "✅" if not action.startswith("Do Nothing") else "⛔"
        print(f"   {icon} {action:<45} {cnt:>5,} คน ({cnt / total_cust * 100:.1f}%)")
    print("-------------------------------------------------------")

    os.makedirs("data/processed", exist_ok=True)
    detail.to_csv("data/processed/final_campaign_engine_output.csv", index=False)
    atomic_write_text("data/processed/model_used.json", json.dumps(
        {"champion_version": ref.version, "sha256": ref.sha256, "provenance": ref.provenance}, indent=2))
    log("L4", "l4_serving_complete", champion_version=ref.version, customers=total_cust,
        send_count=int(detail['should_send'].sum()))

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

    keep_cols = ['customer_id', 'recommended_action', 'expected_profit',
                 'guardrail_blocked', 'should_send', 'send_window']
    if 'cluster' in detail.columns:
        keep_cols.insert(1, 'cluster')
    return detail[keep_cols]


if __name__ == "__main__":
    run_layer4()
