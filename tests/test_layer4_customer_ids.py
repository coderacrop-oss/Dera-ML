from layer4_uplift_margin import _load_real_labels


def _write(tmp_path, ids):
    rows = ["customer_id,timestamp,total_price,is_treated,post_promo_visit,promo_type"]
    rows += [f"{c},2025-01-01 10:00:00,5.0,{i % 2},{i % 2},none" for i, c in enumerate(ids)]
    p = tmp_path / "pos.csv"
    p.write_text("\n".join(rows), encoding="utf-8")
    return str(p)


def test_alphanumeric_customer_ids_are_kept(tmp_path):
    out = _load_real_labels(_write(tmp_path, ["M02073", "N4F5VWP", "M02073"]))
    assert sorted(out["customer_id"].tolist()) == ["M02073", "N4F5VWP"]


def test_float_formatted_numeric_ids_are_normalised(tmp_path):
    out = _load_real_labels(_write(tmp_path, ["123.0", "456.0", "123.0"]))
    assert sorted(out["customer_id"].tolist()) == ["123", "456"]
