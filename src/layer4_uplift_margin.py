import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import pickle
from datetime import datetime

from layer1_valuation import run_layer1
from layer2_bundling import run_layer2
from layer3_context import run_layer3
from business_config import CONFIG

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
    if not os.path.exists(promo_path):
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
        df['customer_id'] = df['customer_id'].astype(str)
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
        # แปลง conversion rate → binary: > median rate = converted
        median_conv = cust.loc[cust['was_treated']==1, 'converted'].median()
        cust['converted'] = (cust['converted'] > median_conv).astype(int)
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


def run_layer4(pos_file_path="data/transactions.csv",
               promo_response_path="data/marketing_campaign.csv"):
    print("=== กำลังประมวลผล Layer 4: Campaign Engine (Dynamic Context Validation Gate) ===")

    if not os.path.exists(pos_file_path):
        print(f"❌ ไม่พบไฟล์ข้อมูล: {pos_file_path}")
        return

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
        # fallback สำหรับลูกค้าที่ไม่มีใน marketing_campaign
        raw_df['was_treated']    = raw_df['was_treated'].fillna(
            (raw_df['rfm_decile_segment'] >= median_decile).astype(int))
        raw_df['converted']      = raw_df['converted'].fillna(
            (raw_df['frequency'] > 1).astype(int))
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
        print("ℹ️  [Layer 4] ใช้ RFM proxy labels (ไม่พบ promo response file)")
        raw_df['was_treated'] = (raw_df['rfm_decile_segment'] >= median_decile).astype(int)
        raw_df['converted']   = (raw_df['frequency'] > 1).astype(int)
        features = ['total_price', 'quantity', 'is_campaign_spike',
                    'rfm_decile_segment', 'frequency']

    # --- Customer-level aggregation ---
    agg_dict = {
        'total_price':        ['mean', 'sum'],
        'quantity':           'mean',
        'is_campaign_spike':  'mean',
        'rfm_decile_segment': 'first',
        'frequency':          'first',
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
                 'is_campaign_spike', 'rfm_decile_segment', 'frequency']
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

    treat_data = raw_df[raw_df['was_treated'] == 1]
    ctrl_data  = raw_df[raw_df['was_treated'] == 0]
    
    model_dir = "models/registry"
    os.makedirs(model_dir, exist_ok=True)
    
    champion_path_treat = os.path.join(model_dir, "champion_treat.pkl")
    champion_path_ctrl = os.path.join(model_dir, "champion_ctrl.pkl")
    
    print("🚀 กำลังเทรนโมเดลใหม่ (Challenger Model)...")
    lgb_challenger_treat = lgb.LGBMClassifier(n_estimators=30, random_state=42, verbose=-1)
    lgb_challenger_ctrl = lgb.LGBMClassifier(n_estimators=30, random_state=42, verbose=-1)
    lgb_challenger_treat.fit(treat_data[features], treat_data['converted'])
    lgb_challenger_ctrl.fit(ctrl_data[features], ctrl_data['converted'])
    
    has_champion = False
    if os.path.exists(champion_path_treat) and os.path.exists(champion_path_ctrl):
        with open(champion_path_treat, 'rb') as f: lgb_champion_treat = pickle.load(f)
        with open(champion_path_ctrl, 'rb') as f: lgb_champion_ctrl = pickle.load(f)
        # Only use champion if it was trained on the same feature set
        champ_n_features = lgb_champion_treat.n_features_
        if champ_n_features == len(features):
            has_champion = True
        else:
            print(f"⚠️  [Layer 4] Champion model has {champ_n_features} features but current data has {len(features)} — skipping champion comparison")

    def evaluate_model_profit(model_t, model_c):
        prob_t = model_t.predict_proba(cust_agg[features])[:, 1]
        prob_c = model_c.predict_proba(cust_agg[features])[:, 1]
        
        temp_df = cust_agg.copy()
        temp_df['uplift_score'] = (prob_t - prob_c).round(4)
        temp_df['prob_control'] = prob_c.round(4)
        # rfm_decile_segment และ frequency มีใน cust_agg แล้ว — merge เฉพาะ columns ที่ยังขาด
        rfm_extra = rfm_df[['customer_id', 'recency', 'monetary', 'kmeans_cluster']].drop_duplicates('customer_id')
        temp_df = temp_df.merge(rfm_extra, on='customer_id', how='left')
        temp_df = temp_df.merge(context_df[['customer_id', 'time_context', 'in_stock']].drop_duplicates('customer_id'), on='customer_id', how='left')
        
        COGS_RATE        = CONFIG["cogs_rate"]
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
            uplift    = float(row['uplift_score'])
            prob_c    = float(row['prob_control'])
            rfm_seg   = row['rfm_decile_segment']
            in_stock  = row['in_stock']
            avg_spent = float(row['avg_spent'])
            is_spike  = float(row['is_campaign_spike'])

            gross_margin = avg_spent * (1 - COGS_RATE)
            p_baseline   = max(0.0, prob_c)
            p_treated    = max(0.0, min(1.0, prob_c + uplift))
            baseline_profit = round(gross_margin * p_baseline, 2)

            # --- Do Nothing cases ---
            if pd.isna(in_stock) or not in_stock:
                return pd.Series(["Do Nothing (Out of Stock)", baseline_profit, 0])
            if avg_spent < low_basket:
                return pd.Series(["Do Nothing (Low Basket Size)", baseline_profit, 0])
            if uplift < neg_uplift:
                return pd.Series(["Do Nothing (Negative Uplift Guard)", baseline_profit, 0])

            # Helper: per-customer CFO Guardrail check
            # Incremental profit = expected profit gain from sending promo vs not sending
            # = gross_margin * (p_treated - p_baseline) - promo_cost
            # If incremental_profit < 0 OR net_margin < floor → Do Nothing
            def _guardrail_ok(promo_cost_rate):
                promo_cost = avg_spent * promo_cost_rate
                gross_rev  = avg_spent * (1 - CONFIG["default_discount_rate"])
                net_profit = gross_rev - avg_spent * COGS_RATE - promo_cost
                net_margin = net_profit / gross_rev if gross_rev > 0 else 0
                incremental = gross_margin * (p_treated - p_baseline) - promo_cost
                return net_margin >= MIN_MARGIN and incremental > 0, promo_cost

            # --- Active campaign cases with per-customer guardrail ---
            if is_spike > 0.5 and rfm_seg >= spike_rfm:
                ok, promo_cost = _guardrail_ok(PROMO_COST_RATES["festival"])
                if not ok:
                    return pd.Series(["Do Nothing (CFO Guardrail)", baseline_profit, 1])
                return pd.Series(["Standard Retention (Festival Guard)",
                                  round(gross_margin * p_treated - promo_cost, 2), 0])

            if avg_spent < mid_basket and rfm_seg >= mid_rfm:
                ok, promo_cost = _guardrail_ok(PROMO_COST_RATES["off_peak"])
                if not ok:
                    return pd.Series(["Do Nothing (CFO Guardrail)", baseline_profit, 1])
                return pd.Series(["Off-Peak Special: ส่วนลดกระตุ้นช่วงบ่าย",
                                  round(gross_margin * p_treated - promo_cost, 2), 0])

            if rfm_seg >= vip_rfm and prob_c > vip_prob:
                ok, promo_cost = _guardrail_ok(PROMO_COST_RATES["vip_points"])
                if not ok:
                    return pd.Series(["Do Nothing (CFO Guardrail)", baseline_profit, 1])
                return pd.Series(["VIP Reward: แจกแต้ม X2",
                                  round(gross_margin * p_treated - promo_cost, 2), 0])

            if rfm_seg >= vip_rfm and rfm_seg >= 9:
                ok, promo_cost = _guardrail_ok(PROMO_COST_RATES["vip_points"])
                if not ok:
                    return pd.Series(["Do Nothing (CFO Guardrail)", baseline_profit, 1])
                return pd.Series(["VIP Early Access",
                                  round(gross_margin * p_treated - promo_cost, 2), 0])

            if uplift > high_uplift:
                ok, promo_cost = _guardrail_ok(PROMO_COST_RATES["bundle"])
                if not ok:
                    return pd.Series(["Do Nothing (CFO Guardrail)", baseline_profit, 1])
                return pd.Series([f"Push Promotion: {top_bundle_name}",
                                  round(gross_margin * p_treated - promo_cost, 2), 0])

            ok, promo_cost = _guardrail_ok(PROMO_COST_RATES["standard"])
            if not ok:
                return pd.Series(["Do Nothing (CFO Guardrail)", baseline_profit, 1])
            return pd.Series(["Standard Retention",
                              round(gross_margin * p_treated - promo_cost, 2), 0])
            
        res = temp_df.apply(calculate_action_and_profit, axis=1)
        temp_df['recommended_action'] = res[0]
        temp_df['expected_profit']    = res[1]
        temp_df['guardrail_blocked']  = res[2]
        temp_df = temp_df.sort_values(by='expected_profit', ascending=False).reset_index(drop=True)
        return temp_df['expected_profit'].sum(), temp_df

    challenger_profit, challenger_detail_df = evaluate_model_profit(lgb_challenger_treat, lgb_challenger_ctrl)
    
    is_passed = True
    growth_pct = 0.0
    
    # 2. Market Context Evaluation Logic (เช็กสภาวะตลาด)
    # Baseline = avg spending of first 75% of timeline; Current = most recent 25%
    raw_df['timestamp'] = pd.to_datetime(raw_df.get('timestamp', pd.NaT), errors='coerce')
    ts_min = raw_df['timestamp'].min()
    ts_max = raw_df['timestamp'].max()
    if pd.notna(ts_min) and pd.notna(ts_max) and ts_min != ts_max:
        cutoff = ts_min + (ts_max - ts_min) * 0.75
        baseline_avg_spent = raw_df.loc[raw_df['timestamp'] <= cutoff, 'total_price'].mean()
        current_avg_spent  = raw_df.loc[raw_df['timestamp'] >  cutoff, 'total_price'].mean()
        # fallback if slice is empty
        if pd.isna(baseline_avg_spent) or baseline_avg_spent <= 0:
            baseline_avg_spent = raw_df['total_price'].mean()
        if pd.isna(current_avg_spent) or current_avg_spent <= 0:
            current_avg_spent = raw_df['total_price'].mean()
    else:
        baseline_avg_spent = raw_df['total_price'].mean()
        current_avg_spent  = baseline_avg_spent
    is_recession_mode = current_avg_spent < (baseline_avg_spent * CONFIG["recession_threshold"])
    
    gate_reason = ""
    
    if has_champion:
        champion_profit, _ = evaluate_model_profit(lgb_champion_treat, lgb_champion_ctrl)
        growth_pct = ((challenger_profit - champion_profit) / champion_profit) * 100
        
        if is_recession_mode:
            # ในภาวะวิกฤต ถ้าโมเดลใหม่รัดกุมกว่า (Guard Actions > 50%) จะอนุมัติเพื่อป้องกันขาดทุน
            do_nothing_ratio = (challenger_detail_df['recommended_action'].str.contains("Do Nothing").mean()) * 100
            dn_thr    = CONFIG["do_nothing_recession_threshold"]
            gate_tgt  = CONFIG["gate_growth_target_pct"]
            if do_nothing_ratio > dn_thr:
                is_passed = True
                gate_reason = f"⚠️ [RECESSION MODE DETECTED]: ตลาดซบเซา (เฉลี่ย {current_avg_spent:,.2f} บาท) -> อนุมัติโมเดลใหม่ที่เน้น Defensive เพื่อเซฟงบ (Do Nothing = {do_nothing_ratio:.1f}%)"
            else:
                is_passed = (growth_pct >= gate_tgt)
                gate_reason = "ตลาดซบเซาแต่โมเดลใหม่ไม่เน้นเซฟงบ"
        else:
            gate_tgt = CONFIG["gate_growth_target_pct"]
            if abs(growth_pct) < 0.5:
                # virtually identical — same data re-run
                is_passed = False
                gate_reason = (
                    f"โมเดลใหม่ให้ผลเท่าเดิม ({growth_pct:+.2f}%) — "
                    f"น่าจะเทรนด้วยข้อมูลชุดเดิม ไม่มี signal ใหม่ คงโมเดลเก่าไว้"
                )
            elif growth_pct < 0:
                is_passed = False
                gate_reason = (
                    f"โมเดลใหม่ประเมินกำไรได้ต่ำกว่าโมเดลเก่า {growth_pct:+.2f}% — "
                    f"อาจเกิดจาก dataset คนละชุด (distribution shift) หรือโมเดลเก่า overoptimistic "
                    f"คงโมเดลเก่าไว้ก่อน แนะนำ: reset champion ถ้าเปลี่ยน dataset ใหม่ทั้งหมด"
                )
            elif growth_pct < gate_tgt:
                is_passed = False
                gate_reason = (
                    f"โมเดลใหม่ดีขึ้นเล็กน้อย {growth_pct:+.2f}% แต่ยังไม่ถึงเกณฑ์ {gate_tgt:.1f}% "
                    f"ที่ตั้งไว้ — คงโมเดลเก่าไว้"
                )
            else:
                is_passed = True
                gate_reason = (
                    f"โมเดลใหม่ดีกว่าโมเดลเก่า {growth_pct:+.2f}% ผ่านเกณฑ์ {gate_tgt:.1f}% — "
                    f"อัปเดต champion"
                )

    # Calculate should_send early so action breakdown can use it
    challenger_detail_df['should_send'] = (
        (challenger_detail_df['guardrail_blocked'] == 0) &
        (~challenger_detail_df['recommended_action'].str.startswith('Do Nothing'))
    ).astype(int)

    # --- Action breakdown summary ---
    action_counts = challenger_detail_df['recommended_action'].value_counts()
    send_count    = int(challenger_detail_df['should_send'].sum())
    total_cust    = len(challenger_detail_df)

    print("\n=======================================================")
    print("      PRODUCTION MODEL VALIDATION GATE REPORT          ")
    print("=======================================================")
    if is_recession_mode:
        print("⚠️  MARKET STATUS: RECESSION — ตลาดซบเซา ใช้กลยุทธ์ป้องกันขาดทุน")
    else:
        print("🟢 MARKET STATUS: NORMAL — ตลาดปกติ โมดเติบโตได้")

    if has_champion:
        diff = challenger_profit - champion_profit
        print(f"\n📊 ผลประเมินโมเดล (บน {total_cust:,} ลูกค้า):")
        print(f"   Champion (โมเดลเก่า) : {champion_profit:>12,.2f} บาท")
        print(f"   Challenger (โมเดลใหม่): {challenger_profit:>12,.2f} บาท  ({growth_pct:+.2f}%)")
        print(f"   ส่วนต่าง             : {diff:>+12,.2f} บาท")
        print(f"\n💡 สรุป: {gate_reason}")
    else:
        print(f"\n📊 ไม่มีโมเดลเก่า — ใช้ผลประเมิน Challenger: {challenger_profit:,.2f} บาท")
        print(f"💡 สรุป: บันทึกโมเดลนี้เป็น Champion ตัวแรก")

    print(f"\n🎯 การตัดสินใจ Layer 4 ({total_cust:,} ลูกค้า):")
    for action, cnt in action_counts.items():
        pct = cnt / total_cust * 100
        icon = "✅" if not action.startswith("Do Nothing") else "⛔"
        print(f"   {icon} {action:<45} {cnt:>5,} คน ({pct:.1f}%)")

    print("-------------------------------------------------------")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("models/archive", exist_ok=True)

    if is_passed:
        with open(champion_path_treat, 'wb') as f: pickle.dump(lgb_challenger_treat, f)
        with open(champion_path_ctrl, 'wb') as f: pickle.dump(lgb_challenger_ctrl, f)
        print(f"🟢 GATE PASSED — อัปเดต Champion เรียบร้อย")
    else:
        print(f"🔴 GATE FAILED — คงใช้ Champion เดิม")
        if has_champion and growth_pct < -5:
            print(f"   ⚠️  หากเพิ่งเปลี่ยน dataset ใหม่ทั้งหมด ให้รัน: rm models/registry/champion_*.pkl แล้วรันใหม่")

    print("=======================================================\n")
    
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/final_campaign_engine_output.csv"
    challenger_detail_df.to_csv(output_path, index=False)

    # Return per-customer detail df so caller can merge into segment plan
    keep_cols = ['customer_id', 'recommended_action', 'expected_profit', 'guardrail_blocked', 'should_send']
    # Include cluster column if present
    if 'cluster' in challenger_detail_df.columns:
        keep_cols.insert(1, 'cluster')
    return challenger_detail_df[keep_cols]

if __name__ == "__main__":
    run_layer4()
