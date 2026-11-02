import pandas as pd
import numpy as np
from datetime import datetime
import os
from business_config import CONFIG

def run_layer3(pos_file_path="data/raw_pos_transactions.csv"):
    print("=== กำลังประมวลผล Layer 3: Dynamic Context (Peak/Off-Peak & Inventory Check) ===")
    df = pd.read_csv(pos_file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    n_bad_ts = df['timestamp'].isna().sum()
    if n_bad_ts > 0:
        print(f"⚠️ [Layer 3] timestamp แปลงไม่ได้ {n_bad_ts} แถว — ข้ามแถวเหล่านั้น")
        df = df.dropna(subset=['timestamp'])

    # 1. กำหนด Peak / Off-Peak จาก Hour (Peak: 17:00 - 21:00)
    # Normalize to business timezone before extracting hour
    biz_tz = CONFIG["business_tz"]
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(biz_tz, ambiguous='infer', nonexistent='shift_forward')
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert(biz_tz)
    peak_start = CONFIG["peak_hour_start"]
    peak_end   = CONFIG["peak_hour_end"]
    df['hour'] = df['timestamp'].dt.hour
    df['time_context'] = df['hour'].apply(lambda h: 'Peak' if peak_start <= h <= peak_end else 'Off-Peak')

    # 2. ประมาณ stock จาก sales velocity (ไม่มี inventory จริง)
    # หลักการ: ร้านสั่งของเข้าทุก restock_cycle_days วัน โดยสั่งตาม avg demand × buffer
    # สิ่งที่อาจ "หมด": สินค้าที่มี demand เร่งตัวขึ้นในช่วงล่าสุด จนเกิน restock estimate
    stock_buffer    = CONFIG["stock_buffer_multiplier"]
    low_stock_thr   = CONFIG["low_stock_threshold"]
    restock_days    = CONFIG.get("stock_restock_cycle_days", 14)

    n_days = max(1, (df['timestamp'].max() - df['timestamp'].min()).days)
    total_qty = df.groupby('product_name')['quantity'].sum()
    daily_rate = total_qty / n_days

    # Estimated stock per restock cycle (how much the shop orders each cycle)
    initial_est = (daily_rate * restock_days * stock_buffer).clip(lower=10).round().astype(int)

    # Recent demand: sales in the last restock_cycle_days window
    recent_cutoff = df['timestamp'].max() - pd.Timedelta(days=restock_days)
    recent_qty = (
        df[df['timestamp'] >= recent_cutoff]
        .groupby('product_name')['quantity'].sum()
        .reindex(total_qty.index, fill_value=0)
    )

    stock_summary = total_qty.reset_index().rename(columns={'quantity': 'total_quantity'})
    stock_summary['initial_stock_est'] = initial_est.values
    stock_summary['recent_demand']     = recent_qty.values
    # Remaining = what was stocked this cycle minus recent demand
    stock_summary['remaining_stock']   = (stock_summary['initial_stock_est'] - stock_summary['recent_demand']).clip(lower=0)
    # Use relative threshold: out of stock if remaining < 20% of estimated cycle demand
    stock_summary['in_stock'] = (
        stock_summary['remaining_stock'] >
        (stock_summary['initial_stock_est'] * 0.20).clip(lower=low_stock_thr)
    )

    # Rename for compatibility with downstream merge
    stock_summary = stock_summary.rename(columns={'total_quantity': 'quantity'})
    
    # 3. สรุป Context ของแต่ละธุรกรรม/ลูกค้า
    context_df = df[['transaction_id', 'customer_id', 'product_name', 'timestamp', 'time_context']].drop_duplicates()
    context_df = context_df.merge(stock_summary[['product_name', 'remaining_stock', 'in_stock']], on='product_name', how='left')
    
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/layer3_context_flags.csv"
    context_df.to_csv(output_path, index=False)
    
    return context_df

if __name__ == "__main__":
    run_layer3()
