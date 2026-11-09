# DERA Bug Fixes — Nov 2026

สรุปสาเหตุและวิธีแก้ไขบั๊กทั้ง 4 รายการที่ DERA รายงาน

---

## #1 — `incremental_profit` คำนวณผิด

**ไฟล์:** `src/uplift_engine.py`

**Root Cause:**
`_viable_loyalty()` ใช้ `gross_margin * p_baseline` แทน `gross_margin * (p_treated − p_baseline)` ทำให้ ROI พองสูงขึ้น ~21× เพราะรวม baseline probability ที่ไม่ใช่ผลจากโปรโมชั่นเข้าไปด้วย
`_viable()` (discount/cashback) และ `_viable_clv()` มีปัญหาเดียวกัน — ใช้ `p_treated` แทน incremental `(p_treated − p_baseline)`

**Fix:**
- เพิ่ม helper `_incremental(rate_key)` ที่คำนวณตามสูตรที่ถูกต้องเพียงจุดเดียว:
  ```
  incremental_profit = gross_margin × (p_treated − p_baseline) − promo_cost
  ```
- `_viable()`, `_viable_loyalty()`, `_viable_clv()` ทุกตัว delegate ไปที่ `_incremental()`
- `select_within_budget` ใช้ `profit_col="incremental_profit"` เป็น default

**คอลัมน์ใหม่ใน output:**
| คอลัมน์ | ความหมาย |
|---------|----------|
| `incremental_profit` | กำไรส่วนเพิ่มที่เกิดจากโปรโมชั่นจริงๆ |
| `gross_margin` | gross margin ที่ใช้คำนวณ |
| `p_baseline` | ความน่าจะเป็นซื้อโดยไม่มีโปรโมชั่น |
| `p_treated` | ความน่าจะเป็นซื้อเมื่อได้รับโปรโมชั่น |

---

## #2 — `uplift_group` และ `decision_reason` ไม่ปรากฏใน output

**ไฟล์:** `src/uplift_engine.py`

**Root Cause:**
`classify_uplift_group()` ถูกเรียกใน `calculate_action_and_profit()` แต่ผลลัพธ์ถูกทิ้ง ไม่ได้ส่งออกมาใน `pd.Series` return value
`guardrail_blocked` ถูก hardcode เป็น `0` เสมอ ไม่มี column ที่บอกว่าลูกค้าถูก block เพราะอะไร

**Fix:**
- เพิ่ม helper `_s()` ใน `calculate_action_and_profit` เพื่อ enforce format ของ return Series ที่มี 10 elements:
  ```
  [action, profit, guardrail_blocked, window, promo_cost,
   uplift_group, decision_reason, gross_margin, p_baseline, p_treated]
  ```
- ทุก return path (รวม hard_stop, do_not_disturb, low_basket) ผ่าน `_s()` เดียวกัน
- `decide_for_customers()` map columns ใหม่:
  ```python
  temp_df['uplift_group']    = res[5]
  temp_df['decision_reason'] = res[6]
  ```

**คอลัมน์ใหม่ใน output:**
| คอลัมน์ | ค่าที่เป็นไปได้ |
|---------|---------------|
| `uplift_group` | `do_not_disturb`, `lost_cause`, `sure_thing`, `persuadable` |
| `decision_reason` | `hard_stop`, `do_not_disturb`, `lost_cause`, `sure_thing_[type]`, `persuadable_[type]`, `low_basket_[...]` |

---

## #3 — Champion v7 metrics ขัดแย้งกัน / AUUC ติดลบไม่ถูก block

**ไฟล์:** `src/candidates.py`

**Root Cause:**
`gate_decision()` ใช้แค่ profit metrics ในการตัดสิน ไม่เคยตรวจ AUUC/Qini
ทำให้โมเดลที่มี `ate_pred = −0.0845` (negative average treatment effect) และ `auuc = −0.1007` (worse than random) ผ่านการ approve ได้ เพราะ holdout profit บังเอิญดูดี

AUUC ติดลบหมายความว่าโมเดล rank ลูกค้าที่จะได้ประโยชน์จากโปรโมชั่น **ผิดทาง** — ส่งให้คนที่ respond น้อยกว่า random ก่อน

**Fix:**
เพิ่ม hard-block หลัง data quality gate ใน `candidates.py`:
```python
if (
    pooled.label_mode == "real"
    and isinstance(uplift_m.get("auuc"), (int, float))
    and uplift_m["auuc"] < 0
    and gate.passed
):
    gate = GateResult(
        passed=False,
        reason=f"❌ [AUUC] AUUC = {uplift_m['auuc']:.4f} (negative) — model performs "
               "worse than random targeting; champion blocked until retrain",
        growth_pct=gate.growth_pct,
    )
```

**ผลที่ได้:** champion approval จะถูก block ใน 2 กรณี:
1. Data quality ไม่ผ่าน (เดิม)
2. AUUC < 0 แม้ profit gate ผ่าน (ใหม่)

---

## #4 — Bug: `cluster_segment_name` column ไม่มีใน DataFrame

**ไฟล์:** `src/promo_advisor.py`

**Root Cause:**
`decide_for_customers()` ใน `uplift_engine.py` ส่ง column ชื่อ `cluster_segment` แต่
`evaluate_operator_promo()` ใน `promo_advisor.py` อ่าน `cluster_segment_name` ทำให้ทุกแถว `promo_eligible = False` และ `recommended_action` ของคนที่ไม่ถูกส่งเป็น empty string

**Fix:**
แก้ 3 จุดใน `promo_advisor.py` ด้วย graceful column detection:

**จุดที่ 1** — `evaluate_operator_promo` segment filter (~line 210):
```python
seg_col = next(
    (c for c in ("cluster_segment", "cluster_segment_name", "tier") if c in df.columns),
    None,
)
if seg_col:
    df["promo_eligible"] = df[seg_col].str.lower().str.strip().isin(seg_lower)
else:
    df["promo_eligible"] = False
```

**จุดที่ 2** — `analyze_promo_history` groupby (~line 633):
```python
seg_col_pos = next(
    (c for c in ("cluster_segment", "cluster_segment_name") if c in pos_df.columns),
    None,
)
if seg_col_pos:
    grp_cols.append(seg_col_pos)
```

**จุดที่ 3** — segment stats groupby (~line 662):
```python
_seg_col = next(
    (c for c in ("cluster_segment", "cluster_segment_name") if c in cust.columns),
    None,
)
if _seg_col:
    for seg, grp in cust.groupby(_seg_col):
        ...
```

---

## สรุป

| # | บั๊ก | ไฟล์ | สถานะ |
|---|------|------|-------|
| 1 | `incremental_profit` ใช้สูตรผิด | `uplift_engine.py` | ✅ แก้แล้ว |
| 2 | `uplift_group` / `decision_reason` ไม่ออก | `uplift_engine.py` | ✅ แก้แล้ว |
| 3 | AUUC ติดลบไม่ block champion | `candidates.py` | ✅ แก้แล้ว |
| 4 | `cluster_segment_name` column ไม่มี | `promo_advisor.py` | ✅ แก้แล้ว |
| 5 | RFM segment ไม่เสถียร | `uplift_engine.py` | 🟢 Backlog |
| 6 | PSM + retrain queue endpoint | `retrain_scheduler.py` | 🟢 Backlog |
