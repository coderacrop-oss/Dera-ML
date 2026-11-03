"""Deterministic small POS CSVs for Layer 4 / training tests."""
import numpy as np
import pandas as pd

PRODUCTS = {"coffee": 4.0, "tea": 3.0, "cake": 6.5, "sandwich": 8.0, "juice": 5.0, "salad": 9.0, "cookie": 2.5, "soup": 7.0}


def make_pos_csv(tmp_path, n_customers=120, seed=0, with_labels=False, name="pos.csv", id_prefix="c") -> str:
    """One row per line item. Columns: transaction_id, customer_id, timestamp, product_name, quantity, total_price
    (+ is_treated, post_promo_visit, promo_type when with_labels=True, one value per customer)."""
    rng = np.random.RandomState(seed)
    names = list(PRODUCTS)
    rows = []
    tx = 0
    for i in range(n_customers):
        cid = f"{id_prefix}{i:03d}"
        treated = int(rng.rand() < 0.45)
        visits = int(rng.rand() < (0.55 if treated else 0.40))
        for _ in range(int(rng.randint(2, 7))):
            tx += 1
            day = int(rng.randint(0, 60))
            hour = int(rng.choice([8, 9, 12, 13, 17, 18, 19, 15, 21]))
            ts = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day, hours=hour, minutes=int(rng.randint(0, 60)))
            for _ in range(int(rng.randint(1, 3))):
                p = names[int(rng.randint(0, len(names)))]
                q = int(rng.randint(1, 4))
                row = dict(transaction_id=f"T{tx:05d}", customer_id=cid, timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
                           product_name=p, quantity=q, total_price=round(q * PRODUCTS[p], 2))
                if with_labels:
                    row.update(is_treated=treated, post_promo_visit=visits, promo_type="promo" if treated else "none")
                rows.append(row)
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)
