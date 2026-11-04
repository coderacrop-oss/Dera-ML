import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn.calibration import CalibratedClassifierCV
import json
import pickle
from dataclasses import dataclass, field
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


class MultiTreatmentUpliftModel:
    """Wrapper for multiple X-learner models — one per treatment type.

    Used when historical data has randomized multi-arm treatment assignment
    (discount / points_vip / bundle / reactivation).

    Selects the treatment type with highest CATE × margin for each customer.
    Falls back to single-model CATE if a treatment type has no model.

    Training: see train_multi_treatment.py
    Data requirement: labels CSV must have 'treatment_type' column with values
    matching the keys of `models` dict.
    """

    TREATMENT_TYPES = ["discount", "points_vip", "bundle", "reactivation"]

    def __init__(self, models: dict):
        # models: {"discount": XLearnerModel, "points_vip": XLearnerModel, ...}
        self.models = models

    def predict_all_cates(self, X) -> dict:
        """Returns {treatment_type: cate_array} for all available models."""
        return {t: m.predict_uplift(X) for t, m in self.models.items()}

    def predict_best_treatment(self, X) -> tuple:
        """Returns (best_treatment_type_array, best_cate_array) per customer."""
        import numpy as np
        all_cates = self.predict_all_cates(X)
        if not all_cates:
            raise ValueError("MultiTreatmentUpliftModel has no sub-models")
        types  = list(all_cates.keys())
        matrix = np.column_stack([all_cates[t] for t in types])  # (n, k)
        best_idx = matrix.argmax(axis=1)
        best_treatment = np.array([types[i] for i in best_idx])
        best_cate      = matrix[np.arange(len(best_idx)), best_idx]
        return best_treatment, best_cate

    def predict_uplift(self, X):
        """Compatibility: returns the best CATE (max across all treatments)."""
        _, best_cate = self.predict_best_treatment(X)
        return best_cate

    def predict_control_proba(self, X):
        """Use the first model's control proba as baseline."""
        first = next(iter(self.models.values()))
        return first.predict_control_proba(X)

    @property
    def n_features_(self):
        first = next(iter(self.models.values()))
        return first.n_features_

    @property
    def feature_names_(self):
        first = next(iter(self.models.values()))
        return getattr(first, 'feature_names_', None)


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
    # CalibratedClassifierCV(isotonic, cv=N) fits an isotonic regression on OOF predictions
    # to map raw scores → calibrated probabilities. Also used in Stage 2 cross-predictions.
    # cv is capped so every fold contains at least 1 example of the minority class.
    _cv_treat = min(5, int(treat_data['converted'].value_counts().min()))
    _cv_ctrl  = min(5, int(ctrl_data['converted'].value_counts().min()))
    _cv_treat = max(2, _cv_treat)
    _cv_ctrl  = max(2, _cv_ctrl)
    mu1 = CalibratedClassifierCV(
        lgb.LGBMClassifier(n_estimators=n_estimators, random_state=42,
                           reg_lambda=1.0, min_child_samples=20, subsample=0.8,
                           colsample_bytree=0.8, verbose=-1),
        method='isotonic', cv=_cv_treat
    )
    mu0 = CalibratedClassifierCV(
        lgb.LGBMClassifier(n_estimators=n_estimators, random_state=42,
                           reg_lambda=1.0, min_child_samples=20, subsample=0.8,
                           colsample_bytree=0.8, verbose=-1),
        method='isotonic', cv=_cv_ctrl
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
def _run_layer1(pos_file_path="data/raw_pos_transactions.csv"):
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
    # Log-transform before scaling: reduces extreme skewness in frequency/monetary
    # (skewness 18-19x on raw RFM causes k=2 to always win silhouette by dominating outliers)
    X_log = np.log1p(rfm[['recency', 'frequency', 'monetary']].values)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_log)
    # Business floor: >= 300 customers → min 3 segments (Champions / Mid / At-Risk at minimum)
    min_k = 3 if len(rfm) >= 300 else 2
    max_k = min(10, len(rfm) - 1)
    best_k, best_score = min_k, -1
    for k in range(min_k, max_k + 1):
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
def _run_layer2(pos_file_path="data/raw_pos_transactions.csv"):
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
def _run_layer3(pos_file_path="data/raw_pos_transactions.csv"):
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

    # --- Format 3: pre-aggregated customer-level labels (customer_id, was_treated, converted) ---
    elif 'was_treated' in df.columns and 'converted' in df.columns:
        cid_map = {c: 'customer_id' for c in df.columns if c.lower() in ('customerid', 'customer_id')}
        df = df.rename(columns=cid_map)
        if 'customer_id' not in df.columns:
            print("⚠️ [Layer 4] ไม่รู้จัก format ของ promo file — ข้าม")
            return None
        df['customer_id'] = df['customer_id'].astype(str).str.strip()
        for _col in ('income', 'total_spend', 'times_accepted'):
            if _col in df.columns:
                df[_col] = pd.to_numeric(df[_col], errors='coerce').fillna(0)
            else:
                df[_col] = 0.0
        result = df[['customer_id', 'was_treated', 'converted',
                     'income', 'total_spend', 'times_accepted']].copy()
        source = "pre-aggregated labels (was_treated + converted)"

    else:
        print("⚠️ [Layer 4] ไม่รู้จัก format ของ promo file — ข้าม")
        return None

    n_treated   = int(result['was_treated'].sum())
    n_converted = int(result['converted'].sum())
    print(f"✅ [Layer 4] Real labels loaded [{source}]: {len(result):,} customers | "
          f"treated={n_treated} ({n_treated/len(result)*100:.1f}%) | "
          f"converted={n_converted} ({n_converted/len(result)*100:.1f}%)")
    return result


# ── Window auto-detection ─────────────────────────────────────────────────────

_MIN_POST_DAYS   = 7    # post-window must have at least this many days for conversion measurement
_MIN_TOTAL_DAYS  = 30   # warn when total data span is below this
_DEFAULT_PRE_FRAC = 0.75  # default pre/post split fraction


def _auto_window_split(ts: pd.Series) -> tuple:
    """Choose a time-based pre/post split that guarantees a minimum post-window.

    Unlike quantile(0.75) on transaction timestamps (which measures 75% of
    *transactions*, not 75% of *time*), this splits on the actual date range so
    the post-window is predictable in days.

    Auto-adjusts the split fraction when the default 75/25 would leave fewer
    than _MIN_POST_DAYS days in the post-window:
      • enough total data  → shift split to guarantee _MIN_POST_DAYS post
      • very short dataset → 50/50 split (best we can do)

    Returns (split_date, span_days, post_days).
    """
    ts_clean = pd.to_datetime(ts, errors="coerce").dropna()
    t_min, t_max = ts_clean.min(), ts_clean.max()
    span_days = max(1, (t_max - t_min).days)

    pre_frac = _DEFAULT_PRE_FRAC
    if span_days * (1 - pre_frac) < _MIN_POST_DAYS:
        if span_days >= _MIN_POST_DAYS * 2:
            # Shift split leftward to guarantee minimum post-window
            pre_frac = (span_days - _MIN_POST_DAYS) / span_days
        else:
            pre_frac = 0.5  # very short: 50/50 is the best we can do

    split_date = t_min + pd.Timedelta(days=int(span_days * pre_frac))
    post_days  = max(1, (t_max - split_date).days)
    pre_days   = (split_date - t_min).days

    adj_note = (f" (adjusted from {_DEFAULT_PRE_FRAC:.0%}: post-window too short)"
                if pre_frac != _DEFAULT_PRE_FRAC else "")
    icon = "⚠️ " if span_days < _MIN_TOTAL_DAYS else "ℹ️ "
    print(f"{icon} [Window] {span_days}d span ({t_min.date()} → {t_max.date()}) | "
          f"split: {split_date.date()} | pre: {pre_days}d / post: {post_days}d{adj_note}")
    if span_days < _MIN_TOTAL_DAYS:
        print(f"   Short window (<{_MIN_TOTAL_DAYS} days): uplift signal may be weak — "
              f"consider collecting more history before retraining")

    return split_date, span_days, post_days


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
    cluster_bundle_map: dict = field(default_factory=dict)   # {cluster_id(int): bundle_str}
    cluster_segment_map: dict = field(default_factory=dict)  # {cluster_id(int): segment_name}


def build_serving_context(pos_file_path, promo_response_path=None, product_cogs_dict=None, default_cogs=None):
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

    rfm_df = _run_layer1(norm_path)
    rfm_df['customer_id'] = rfm_df['customer_id'].astype(str)
    bundles_df = _run_layer2(norm_path)
    context_df = _run_layer3(norm_path)
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
        _split, _, _ = _auto_window_split(_ts)
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
        # Use only transaction-derivable features — serve-compatible regardless of labels.
        # promo_dummy_cols excluded: past treatment type is not available at serve time
        # (you're deciding what to send now) and including it conflates treatment
        # assignment with covariates, leaking causal information into features.
        features = ['total_price', 'quantity', 'is_campaign_spike',
                    'rfm_decile_segment', 'frequency']
    else:
        # Natural-experiment proxy: แทน random treatment assignment
        # ────────────────────────────────────────────────────────────────────
        # was_treated = customer ซื้อในช่วง pre-split บนวันที่มี traffic สูง (top 25%)
        # วันที่มี traffic สูง = proxy สำหรับช่วงที่ร้านมีโปรโมชั่น/กิจกรรมจริง
        # converted   = customer กลับมาซื้อในช่วง 25% สุดท้าย (temporal retention)
        #
        # ข้อดีเทียบกับ random:
        #   - มี signal จริง: peak-day shoppers behave differently from off-peak shoppers
        #   - X-learner propensity weighting ช่วย partial-correct selection bias
        # ข้อจำกัด: ยังเป็น observational (self-selection) ไม่ใช่ causal จริง
        # ────────────────────────────────────────────────────────────────────
        print("ℹ️  [Layer 4] Natural-experiment proxy (ไม่พบ real promo labels) — peak-day as treatment")

        _ts = pd.to_datetime(raw_df['timestamp'], errors='coerce')
        _split_date, _span_days, _post_days = _auto_window_split(_ts)

        _pre  = raw_df[_ts <= _split_date].groupby('customer_id')['total_price'].count().rename('n_pre')
        _post = raw_df[_ts >  _split_date].groupby('customer_id')['total_price'].count().rename('n_post')
        _retention = _pre.to_frame().join(_post, how='outer').fillna(0).reset_index()
        _retention['converted'] = (_retention['n_post'] > 0).astype(int)

        # Natural experiment treatment: ซื้อบนวันที่ traffic สูง (top-quartile volume days)
        _pre_ts   = _ts[_ts <= _split_date]
        _pre_rows = raw_df.loc[_pre_ts.index, ['customer_id']].copy()
        _pre_rows['_date'] = _pre_ts.dt.date.values
        _daily_vol = _pre_rows['_date'].value_counts()
        _vol_threshold = _daily_vol.quantile(0.75)
        _campaign_dates = set(_daily_vol[_daily_vol >= _vol_threshold].index)
        _treated_custs = set(
            _pre_rows.loc[_pre_rows['_date'].isin(_campaign_dates), 'customer_id'].unique()
        )
        _retention['was_treated'] = _retention['customer_id'].isin(_treated_custs).astype(int)

        # Safety fallback: ถ้า treatment rate < 10% หรือ > 90% → ไม่มี signal พอ → ใช้ random
        _treat_rate = _retention['was_treated'].mean()
        if not (0.10 <= _treat_rate <= 0.90):
            _rng = np.random.default_rng(42)
            _retention['was_treated'] = _rng.binomial(1, 0.4, len(_retention))
            _treat_rate = _retention['was_treated'].mean()
            print("⚠️  [Layer 4] Peak-day treatment rate out of range → fallback to random assignment")

        raw_df = raw_df.merge(
            _retention[['customer_id', 'converted', 'was_treated']],
            on='customer_id', how='left'
        )
        raw_df['was_treated'] = raw_df['was_treated'].fillna(0).astype(int)
        raw_df['converted']   = raw_df['converted'].fillna(0).astype(int)

        _n_cust  = raw_df['customer_id'].nunique()
        _n_conv  = int(_retention['converted'].sum())
        _n_treat = int(_retention['was_treated'].sum())
        print(f"ℹ️  [Layer 4] split={_split_date.date()} | peak_days={len(_campaign_dates)} | "
              f"customers={_n_cust} | converted={_n_conv} ({_n_conv/_n_cust*100:.1f}%) | "
              f"treated={_n_treat} ({_n_treat/_n_cust*100:.1f}%)")
        print("⚠️  [Layer 4] Proxy mode: uplift scores = observational CATE "
              "(peak-day natural experiment — confounded, not fully causal)")

        # Same base feature set as real-labels path — serve-compatible (derivable from transactions)
        # was_treated is now random binomial so including rfm_decile_segment is safe (no leakage)
        features = ['total_price', 'quantity', 'is_campaign_spike', 'rfm_decile_segment', 'frequency']

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

    # ── Frequency trend: buying more or less recently? ───────────────────────
    # Split window at midpoint; freq_trend = recent_half_count - early_half_count.
    # Positive = rising (buying more often recently), Negative = declining.
    _ts_all = pd.to_datetime(raw_df['timestamp'], errors='coerce')
    _win_mid = _ts_all.min() + (_ts_all.max() - _ts_all.min()) / 2
    _freq_early  = (raw_df[_ts_all <= _win_mid]
                    .groupby('customer_id')['transaction_id'].count()
                    .rename('_freq_early'))
    _freq_recent = (raw_df[_ts_all >  _win_mid]
                    .groupby('customer_id')['transaction_id'].count()
                    .rename('_freq_recent'))
    _freq_parts = _freq_early.to_frame().join(_freq_recent, how='outer').fillna(0)
    _freq_parts['freq_trend'] = (_freq_parts['_freq_recent'] - _freq_parts['_freq_early']).astype(int)
    cust_agg = cust_agg.merge(_freq_parts[['freq_trend']].reset_index(), on='customer_id', how='left')
    cust_agg['freq_trend'] = cust_agg['freq_trend'].fillna(0).astype(int)

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


def classify_uplift_group(uplift: float, prob_control: float) -> str:
    """
    Step 1 of the 2-step decision: classify customer into uplift group.

    Returns one of:
      "persuadable"    — positive uplift, prob_control low-medium → send with promo
      "sure_thing"     — high prob_control, converts regardless → send loyalty only
      "do_not_disturb" — negative uplift, sending hurts conversion → do not send
      "lost_cause"     — near-zero uplift + low baseline → ROI negative for any action

    Thresholds are read from CONFIG.
    """
    neg_guard = CONFIG["negative_uplift_guard"]
    sure_prob = CONFIG["sure_thing_prob_threshold"]
    lc_uplift = CONFIG["lost_cause_uplift_threshold"]
    lc_prob   = CONFIG["lost_cause_prob_threshold"]

    if uplift < neg_guard:
        return "do_not_disturb"
    if prob_control >= sure_prob:
        return "sure_thing"
    if uplift < lc_uplift and prob_control < lc_prob:
        return "lost_cause"
    return "persuadable"


def build_action_menu(
    group: str,
    avg_spent: float,
    cogs: float,
    uplift: float,
    prob_control: float,
    rfm_seg,
    is_spike: float,
    time_context: str,
    top_bundle_name: str = "",
    product_cogs: dict = None,
    default_cogs: float = None,
    cluster_segment_name: str = "",
    bundle_name: str = "",
    rescue: bool = False,
) -> tuple[str, float, str]:
    """
    Step 2 of the 2-step decision: select the best profitable action for customers
    who passed Step 1 (persuadable or sure_thing).

    Rules are defined declaratively in src/action_rules.py.

    Two strategies:
      sure_thing   — SURE_THING_RULES evaluated in order; first profitable action wins.
                     Loyalty-only menu (vip_points / early_access); no discounts.
      persuadable  — PERSUADABLE_RULES: all matching viable rules collected; highest
                     expected profit wins.

    Returns (action_name, expected_profit, send_window).
    Returns ("", baseline_profit, time_context) if no profitable action found.
    """
    from action_rules import SURE_THING_RULES, PERSUADABLE_RULES, cluster_tier, matches

    if not bundle_name:
        bundle_name = top_bundle_name

    PROMO_COST_RATES = CONFIG["promo_cost_rates"]
    MIN_MARGIN       = CONFIG["min_allowable_net_margin"]

    is_off_peak     = (time_context == "Off-Peak")
    send_window     = "Off-Peak Window" if is_off_peak else "Peak Window"
    gross_margin    = avg_spent * (1 - cogs)
    p_baseline      = max(0.0, prob_control)
    p_treated       = max(0.0, min(1.0, prob_control + uplift))
    baseline_profit = round(gross_margin * p_baseline, 2)

    def _viable(rate_key: str) -> tuple[bool, float, float]:
        cost_rate   = PROMO_COST_RATES[rate_key]
        promo_cost  = avg_spent * cost_rate
        gross_rev   = avg_spent * (1 - CONFIG["default_discount_rate"])
        net_profit  = gross_rev - avg_spent * cogs - promo_cost
        net_margin  = net_profit / gross_rev if gross_rev > 0 else 0
        incremental = gross_margin * (p_treated - p_baseline) - promo_cost
        profit      = round(gross_margin * p_treated - promo_cost, 2)
        return (net_margin >= MIN_MARGIN and incremental > 0), profit, round(promo_cost, 2)

    def _viable_loyalty(rate_key: str) -> tuple[bool, float, float]:
        """Sure Things convert regardless — don't require incremental > 0.
        Only check that the promo cost doesn't eliminate baseline profit."""
        cost_rate  = PROMO_COST_RATES[rate_key]
        promo_cost = avg_spent * cost_rate
        profit     = round(gross_margin * p_baseline - promo_cost, 2)
        return profit > 0, profit, round(promo_cost, 2)

    tier = cluster_tier(cluster_segment_name)

    # ── Sure Things: first viable rule wins ──────────────────────────────────
    if group == "sure_thing":
        for rule in SURE_THING_RULES:
            if matches(rule, tier, is_off_peak, rfm_seg, is_spike, uplift):
                ok, profit, promo_cost = _viable_loyalty(rule["rate_key"])
                if ok:
                    action = rule["action"].format(bundle=bundle_name)
                    return action, profit, promo_cost, send_window
        return "", baseline_profit, 0.0, time_context

    # ── Persuadables: collect all viable, return max profit ──────────────────
    # rescue=True (churn_risk / cant_lose / hibernating / about_to_sleep with mildly negative
    # uplift): goal is relationship retention, not incremental conversion — use loyalty check.
    viable_fn = _viable_loyalty if rescue else _viable
    candidates: list[tuple[str, float, float, str]] = []
    for rule in PERSUADABLE_RULES:
        if matches(rule, tier, is_off_peak, rfm_seg, is_spike, uplift):
            ok, profit, promo_cost = viable_fn(rule["rate_key"])
            if ok:
                action = rule["action"].format(bundle=bundle_name)
                candidates.append((action, profit, promo_cost, send_window))

    if candidates:
        return max(candidates, key=lambda x: x[1])

    return "", baseline_profit, 0.0, time_context


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

    # Look up cluster-specific bundle and segment name from context maps
    _cluster_bundle_map  = getattr(ctx, 'cluster_bundle_map', {}) or {}
    _cluster_segment_map = getattr(ctx, 'cluster_segment_map', {}) or {}
    temp_df['cluster_bundle']  = temp_df['kmeans_cluster'].map(_cluster_bundle_map).astype(object).fillna(top_bundle_name)
    temp_df['cluster_segment'] = temp_df['kmeans_cluster'].map(_cluster_segment_map).astype(object).fillna('')

    # ── Rule-based segment flags (override K-Means tier for critical states) ──
    # Applied after K-Means so these always exist even if training data lacked examples.
    # Priority (highest first): cant_lose > churn_risk > new_customer
    _new_max_days   = CONFIG.get("new_customer_max_days",    30)
    _new_max_freq   = CONFIG.get("new_customer_max_freq",     2)
    _churn_days     = CONFIG.get("churn_risk_min_days",      45)
    _churn_freq     = CONFIG.get("churn_risk_min_freq",       3)
    _cantlose_days  = CONFIG.get("cant_lose_min_days",       CONFIG.get("hibernating_recency_days", 90))
    _cantlose_freq  = CONFIG.get("cant_lose_min_freq",        5)
    _cantlose_rfm   = CONFIG.get("cant_lose_min_rfm_decile",  7)
    _sleep_min_days = CONFIG.get("about_to_sleep_min_days", 25)
    _sleep_max_days = CONFIG.get("about_to_sleep_max_days", 45)
    _sleep_max_freq = CONFIG.get("about_to_sleep_max_freq",  4)

    def _apply_segment_flags(row):
        r   = row.get('recency', 999)
        f   = row.get('frequency', 1)
        d   = row.get('rfm_decile_segment', 1)
        seg = row.get('cluster_segment', '')
        ft  = row.get('freq_trend', 0)   # positive = rising, negative = declining

        if r > _cantlose_days and f >= _cantlose_freq and d >= _cantlose_rfm:
            return "Can't Lose Them"
        # Declining customers flagged as Churn Risk sooner (lower recency threshold by 30%)
        _churn_days_eff = _churn_days if ft >= 0 else max(int(_churn_days * 0.70), 25)
        if r > _churn_days_eff and f >= _churn_freq:
            return "Churn Risk"
        # Declining customers flagged as About to Sleep sooner (+10-day window)
        _sleep_max_eff = _sleep_max_days if ft >= 0 else _sleep_max_days + 10
        if _sleep_min_days < r <= _sleep_max_eff and f <= _sleep_max_freq:
            return "About to Sleep"
        if r <= _new_max_days and f <= _new_max_freq:
            return "New Customer"
        return seg

    _n_declining = int((temp_df['freq_trend'] < 0).sum())
    _n_rising    = int((temp_df['freq_trend'] > 0).sum())
    print(f"📈 [Freq Trend] rising={_n_rising} | declining={_n_declining} | "
          f"stable={len(temp_df) - _n_rising - _n_declining}")

    temp_df['cluster_segment'] = temp_df.apply(_apply_segment_flags, axis=1)
    flagged = {k: v for k, v in temp_df['cluster_segment'].value_counts().items()
               if k in ("Can't Lose Them", "Churn Risk", "About to Sleep", "New Customer")}
    if flagged:
        print(f"🚩 [Flags] Rule-based overrides: { {k: v for k, v in flagged.items()} }")

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

    # Data-adaptive basket thresholds
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

    def calculate_action_and_profit(row):
        uplift          = float(row['uplift_score'])
        prob_c          = float(row['prob_control'])
        rfm_seg         = row['rfm_decile_segment']
        in_stock        = row['in_stock']
        avg_spent       = float(row['avg_spent'])
        is_spike        = float(row['is_campaign_spike'])
        time_context    = row.get('time_context', 'Peak')
        top_prod        = row.get('top_product', '')
        cluster_bundle  = row.get('cluster_bundle', top_bundle_name)
        cluster_segment = row.get('cluster_segment', '')
        cogs            = _product_cogs.get(top_prod, _default_cogs)
        gross_margin    = avg_spent * (1 - cogs)
        p_baseline      = max(0.0, prob_c)
        baseline_profit = round(gross_margin * p_baseline, 2)

        # ── Step 1: Hard stops ────────────────────────────────────────────────
        if pd.isna(in_stock) or not in_stock:
            return pd.Series(["Do Nothing (Out of Stock)", baseline_profit, 0, time_context, 0.0])
        if avg_spent < low_basket:
            from action_rules import cluster_tier as _cluster_tier
            tier = _cluster_tier(cluster_segment)

            if tier == "champions":
                m_basket = CONFIG["milestone_basket"]
                m_pts    = CONFIG["milestone_points"]
                m_margin = m_basket * (1 - cogs)
                m_cost   = m_basket * CONFIG["promo_cost_rates"]["milestone_reward"]
                p_mile   = max(0.0, min(1.0, prob_c + uplift * 0.5))
                if m_margin * (p_mile - p_baseline) - m_cost > 0:
                    m_profit = round(m_margin * p_mile - m_cost, 2)
                    return pd.Series([
                        f"Milestone Reward: สั่งครบ {m_basket:.0f} บาท รับ {m_pts} แต้ม",
                        m_profit, 0, time_context, round(m_cost, 2)
                    ])
                return pd.Series(["Do Nothing (Low Basket Size)", baseline_profit, 0, time_context, 0.0])

            elif tier == "new_customer":
                w_cost   = avg_spent * CONFIG["promo_cost_rates"]["new_customer_welcome"]
                w_profit = round(gross_margin * p_baseline - w_cost, 2)
                if w_profit > 0:
                    return pd.Series(["Welcome: สั่งครั้งที่ 2 รับส่วนลด 10%", w_profit, 0, time_context, round(w_cost, 2)])
                return pd.Series(["Do Nothing (Low Basket Size)", baseline_profit, 0, time_context, 0.0])

            elif tier == "about_to_sleep":
                r_cost   = avg_spent * CONFIG["promo_cost_rates"]["standard"]
                r_profit = round(gross_margin * p_baseline - r_cost, 2)
                if r_profit > 0:
                    return pd.Series(["เราคิดถึงคุณ: กลับมาหาเราก่อนที่จะไม่ได้เจอกัน", r_profit, 0, time_context, round(r_cost, 2)])
                return pd.Series(["Do Nothing (Low Basket Size)", baseline_profit, 0, time_context, 0.0])

            elif tier == "hibernating":
                r_cost   = avg_spent * CONFIG["promo_cost_rates"]["bundle_upsell_low"]
                r_profit = round(gross_margin * p_baseline - r_cost, 2)
                if r_profit > 0:
                    return pd.Series(["Reactivation Nudge: กลับมาช้อปรับส่วนลดพิเศษ", r_profit, 0, time_context, round(r_cost, 2)])
                return pd.Series(["Do Nothing (Low Basket Size)", baseline_profit, 0, time_context, 0.0])

            elif tier in ("churn_risk", "cant_lose"):
                r_cost   = avg_spent * CONFIG["promo_cost_rates"]["churn_rescue"]
                r_profit = round(gross_margin * p_baseline - r_cost, 2)
                if r_profit > 0:
                    return pd.Series(["Early Win-Back: กลับมาหาเรา รับส่วนลดพิเศษ", r_profit, 0, time_context, round(r_cost, 2)])
                return pd.Series(["Do Nothing (Low Basket Size)", baseline_profit, 0, time_context, 0.0])

            else:
                bundle_basket = avg_spent * 1.8
                bundle_margin = bundle_basket * (1 - cogs)
                bundle_cost   = bundle_basket * CONFIG["promo_cost_rates"]["bundle_upsell_low"]
                p_upsell      = max(0.0, min(1.0, prob_c + uplift * 0.4))
                if bundle_margin * (p_upsell - p_baseline) - bundle_cost > 0 and cluster_bundle and not cluster_bundle.startswith("เซตคู่คุ้มค่า"):
                    upsell_profit = round(bundle_margin * p_upsell - bundle_cost, 2)
                    return pd.Series([f"Bundle Upsell: เพิ่ม {cluster_bundle}", upsell_profit, 0, time_context, round(bundle_cost, 2)])
                return pd.Series(["Do Nothing (Low Basket Size)", baseline_profit, 0, time_context, 0.0])

        # ── Step 1: Uplift group classification ───────────────────────────────
        from action_rules import cluster_tier as _cluster_tier
        tier = _cluster_tier(cluster_segment)
        group = classify_uplift_group(uplift, prob_c)
        rescue = False
        if group == "do_not_disturb":
            _rescue_tiers = {"churn_risk", "cant_lose", "hibernating"}
            _rescue_guard = CONFIG.get("rescue_uplift_guard", -0.10)
            if tier in _rescue_tiers and uplift >= _rescue_guard:
                group = "persuadable"
                rescue = True
            elif tier in ("new_customer", "potential_loyalist") and prob_c >= CONFIG["sure_thing_prob_threshold"]:
                group = "sure_thing"
            elif tier == "about_to_sleep" and uplift >= _rescue_guard:
                group = "persuadable"
                rescue = True
            else:
                return pd.Series(["Do Nothing (Do Not Disturb)", baseline_profit, 0, time_context, 0.0])
        if group == "lost_cause":
            return pd.Series(["Do Nothing (Lost Cause)", baseline_profit, 0, time_context, 0.0])

        # ── Step 2: Action menu (pre-validated profitable actions only) ───────
        action, profit, promo_cost, send_window = build_action_menu(
            group=group, avg_spent=avg_spent, cogs=cogs,
            uplift=uplift, prob_control=prob_c,
            rfm_seg=rfm_seg, is_spike=is_spike,
            time_context=time_context,
            top_bundle_name=top_bundle_name,
            product_cogs=_product_cogs,
            default_cogs=_default_cogs,
            cluster_segment_name=cluster_segment,
            bundle_name=cluster_bundle,
            rescue=rescue,
        )

        if not action:
            _zc_rfm = CONFIG.get("zero_cost_rfm_threshold", 7)
            _rescue_guard = CONFIG.get("rescue_uplift_guard", -0.10)
            _zc_tiers = {"champions", "loyal", "cant_lose", "potential_loyalist"}
            if rfm_seg >= _zc_rfm and tier in _zc_tiers and uplift >= _rescue_guard:
                return pd.Series(["Zero-Cost: เราคิดถึงคุณ — ขอบคุณที่เป็นลูกค้าที่ดีของเรา",
                                   baseline_profit, 0, time_context, 0.0])
            return pd.Series(["Do Nothing (No Profitable Action)", baseline_profit, 0, time_context, 0.0])

        return pd.Series([action, profit, 0, send_window, promo_cost])

    res = temp_df.apply(calculate_action_and_profit, axis=1)
    temp_df['recommended_action'] = res[0]
    temp_df['expected_profit']    = res[1]
    temp_df['guardrail_blocked']  = res[2]
    temp_df['send_window']        = res[3]
    temp_df['promo_cost']         = res[4]
    temp_df = temp_df.sort_values(by='expected_profit', ascending=False).reset_index(drop=True)
    temp_df['should_send'] = (
        (temp_df['guardrail_blocked'] == 0) &
        (~temp_df['recommended_action'].str.startswith('Do Nothing'))
    ).astype(int)

    # ── Change 5: Treatment type logging ─────────────────────────────────────
    def _treatment_type(action: str) -> str:
        if action.startswith('Do Nothing'):
            return 'none'
        a = action.lower()
        if 'ลด' in a or 'discount' in a:
            return 'discount'
        if 'แต้ม' in a or 'points' in a or 'loyalty' in a or 'vip' in a:
            return 'points_vip'
        if 'reactivation' in a or 'กลับมา' in a:
            return 'reactivation'
        if 'bundle' in a:
            return 'bundle'
        if 'zero-cost' in a or 'ขอบคุณ' in a:
            return 'zero_cost'
        return 'other'

    temp_df['treatment_type'] = temp_df['recommended_action'].apply(_treatment_type)

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


# Pickle compatibility: models saved before the rename still reference 'layer4_uplift_margin'
import sys as _sys
_sys.modules['layer4_uplift_margin'] = _sys.modules[__name__]
