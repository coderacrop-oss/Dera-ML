import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os

def run_layer2(pos_file_path="data/raw_pos_transactions.csv"):
    print("=== กำลังประมวลผล Layer 2: Preference & Bundling (Market Basket Analysis) ===")
    df = pd.read_csv(pos_file_path)
    
    # 1. แปลงข้อมูลให้อยู่ในรูป Basket Matrix (Transaction-Product Matrix)
    # จำกัดเฉพาะ top 150 products ที่ขายบ่อยที่สุด เพื่อให้ Apriori ทำงานได้บน dataset ใหญ่
    TOP_N_PRODUCTS = 150
    top_products = df['product_name'].value_counts().head(TOP_N_PRODUCTS).index
    df_filtered = df[df['product_name'].isin(top_products)]

    n_invoices = df_filtered['transaction_id'].nunique()
    # ปรับ min_support ให้เหมาะกับขนาดข้อมูล: ยิ่งมาก invoice ยิ่งต้องสูงกว่า
    if n_invoices > 5000:
        min_sup = 0.05
    elif n_invoices > 1000:
        min_sup = 0.02
    else:
        min_sup = 0.01

    basket = pd.crosstab(df_filtered['transaction_id'], df_filtered['product_name'])
    basket = basket.map(lambda x: 1 if x > 0 else 0)

    # 2. รัน Apriori เพื่อหา Frequent Itemsets
    frequent_itemsets = apriori(basket, min_support=min_sup, use_colnames=True)
    
    if frequent_itemsets.empty:
        print("คำเตือน: ไม่พบ Itemset ที่ผ่านเกณฑ์ Support ขั้นต่ำ ลองปรับค่า min_support")
        return None

    # 3. สร้าง Association Rules คำนวณ Lift และ Confidence
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    if rules.empty:
        print("คำเตือน: ไม่พบ Association Rules ที่ตรงตามเงื่อนไข")
        return None

    rules = rules.sort_values(by='lift', ascending=False)
    
    top_bundles = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(3)
    
    top_bundles['antecedents'] = top_bundles['antecedents'].apply(lambda x: list(x)[0] if len(x) == 1 else str(x))
    top_bundles['consequents'] = top_bundles['consequents'].apply(lambda x: list(x)[0] if len(x) == 1 else str(x))
    
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/layer2_top_bundles.csv"
    top_bundles.to_csv(output_path, index=False)
    print(f"บันทึกคู่สินค้า Cross-Selling (Top 3 Bundles) สำเร็จ: {output_path}")
    print("\n--- ตัวอย่าง Top 3 Product Bundles ---")
    print(top_bundles)
    
    return top_bundles

if __name__ == "__main__":
    run_layer2()
