import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def run_layer1(pos_file_path):
    df = pd.read_csv(pos_file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate RFM
    max_date = df['timestamp'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_id').agg({
        'timestamp': lambda x: (max_date - x.max()).days, # Recency
        'transaction_id': 'count',                       # Frequency
        'total_price': 'sum'                             # Monetary
    }).reset_index()
    
    rfm.columns = ['customer_id', 'Recency', 'Frequency', 'Monetary']
    
    # Scale Data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    # Dynamic k via Silhouette Score (k=2..5)
    best_k, best_score = 2, -1
    max_k = min(5, len(rfm) - 1)
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(rfm_scaled)
        score = silhouette_score(rfm_scaled, labels)
        if score > best_score:
            best_score, best_k = score, k

    rfm['Cluster'] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(rfm_scaled)

    print(f"=== Layer 1: RFM Clustering Complete (k={best_k}, Silhouette={best_score:.3f}) ===")
    print(rfm.groupby('Cluster').mean())
    return rfm

if __name__ == "__main__":
    rfm_result = run_layer1("data/raw_pos_transactions.csv")
    rfm_result.to_csv("output/layer1_customer_segments.csv", index=False)
