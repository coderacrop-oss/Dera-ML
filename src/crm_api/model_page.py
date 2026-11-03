"""HTML for the champion-approval page. Pure function; every value is HTML-escaped."""
from __future__ import annotations

import json
from html import escape as _e


def _n(x) -> str:
    return "-" if x is None else f"{x:,.2f}"


def _i(x) -> str:
    return "-" if x is None else f"{int(x):,}"


def _js(value) -> str:
    """A JSON literal that is safe inside a <script> element."""
    return json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _col(title: str, m: dict | None, growth: float | None = None) -> str:
    if m is None:
        return f"<div class='col'><h3>{_e(title)}</h3><p>ไม่มี / เทียบไม่ได้</p></div>"
    g = "" if growth is None else f" <span class='delta'>({growth:+.2f}%)</span>"
    return (f"<div class='col'><h3>{_e(title)}</h3>"
            f"<p>กำไรที่ประมาณได้<br><b>฿ {_n(m['profit'])}</b>{g}</p>"
            f"<p>ลูกค้าที่จะส่งโปร<br><b>{_i(m['send_count'])} คน</b></p>"
            f"<p>ถูก CFO Guardrail บล็อก<br><b>{_i(m['blocked_count'])} คน</b></p></div>")


def render_candidate_page(candidate_id: str, report: dict, status: dict, current_version: str | None, token: str) -> str:
    r = report
    stale = r.get("compared_against_version") != current_version
    pending = status.get("status") == "pending"
    training = r.get("training") or {}
    files = "<br>".join(f"{_e(f['store_id'])}: {_e(f['csv'])}" + (f" + {_e(f['labels'])}" if f.get("labels") else "")
                        for f in training.get("files", []))
    dr = training.get("date_range") or ["-", "-"]
    store_rows = "".join(
        f"<tr><td>{_e(s['store_id'])}</td><td>{_i(s['rows'])}</td><td>{_i(s['customers'])}</td>"
        f"<td>{_i(s['holdout_customers'])}</td></tr>" for s in r["stores"])
    cautions = []
    if r["label_mode"] != "real":
        cautions.append("ใช้ label แบบประมาณการ (proxy: สุ่มกลุ่มได้รับโปร) — ผลเปรียบเทียบนี้ไม่ใช่หลักฐานว่าโปรโมชันได้ผลจริง")
    if r["holdout_warning"]:
        cautions.append(f"ลูกค้าที่กันไว้ประเมินมีเพียง {_i(r['holdout_size'])} คน (น้อยกว่า 100) ตัวเลขอาจไม่นิ่ง")
    cautions.append("ค่าสถิติรวม (RFM, เกณฑ์ราคา) คำนวณจากข้อมูลทั้งไฟล์ตามที่ใช้ตอนใช้งานจริง ไม่ได้แยกเฉพาะชุดเทรน")
    cautions.append("กำไรเป็นค่าประมาณจากโมเดลโดยใช้ต้นทุนสินค้า (COGS) ค่าเริ่มต้น ไม่ใช่รายได้จริง")
    caution_html = "".join(f"<li>{_e(c)}</li>" for c in cautions)
    gate = r["gate"]
    badge = "ผ่านเกณฑ์" if gate["passed"] else "ไม่ผ่านเกณฑ์"
    cmp_v = r.get("compared_against_version")
    cur = "ยังไม่มี" if cmp_v is None else _e(cmp_v)

    if not pending:
        action = (f"<p class='state'>ตัดสินใจแล้ว: <b>{_e(status['status'])}</b> โดย {_e(status.get('approver') or '-')} "
                  f"เมื่อ {_e(status.get('decided_at') or '-')}</p>")
    else:
        stale_html = ("<p class='stale'>ผู้สมัครนี้ล้าสมัย: Champion ปัจจุบันเปลี่ยนไปแล้วหลังจากเทียบ "
                      "ต้องเทรนใหม่ จึงอนุมัติไม่ได้</p>") if stale else ""
        approve_btn = "" if stale else "<button id='approve' class='ok'>อนุมัติ → ตั้งเป็น Champion</button>"
        action = (f"{stale_html}<label>ชื่อผู้อนุมัติ <input id='approver' name='approver' maxlength='80'></label>"
                  f"<div class='btns'>{approve_btn}<button id='reject'>ไม่อนุมัติ (คง Champion เดิม)</button></div>"
                  f"<p id='result' class='state'></p>"
                  f"<script>const CID={_js(candidate_id)},TOKEN={_js(token)};"
                  "async function act(a){const n=document.getElementById('approver').value.trim();"
                  "if(!n){document.getElementById('result').textContent='กรุณากรอกชื่อผู้อนุมัติ';return;}"
                  "const r=await fetch('/v1/models/candidates/'+CID+'/'+a,{method:'POST',headers:{'Content-Type':'application/json'},"
                  "body:JSON.stringify({token:TOKEN,approver:n})});const j=await r.json();"
                  "document.getElementById('result').textContent=r.ok?('สำเร็จ: '+(j.version||j.status)):((j.error&&j.error.message)||'ผิดพลาด');"
                  "if(r.ok)setTimeout(()=>location.reload(),800);}"
                  "const a=document.getElementById('approve');if(a)a.onclick=()=>act('approve');"
                  "document.getElementById('reject').onclick=()=>act('reject');</script>")

    return f"""<!doctype html><html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>อนุมัติ Champion — {_e(candidate_id)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;line-height:1.5}}
.cols{{display:flex;gap:1rem;flex-wrap:wrap}}.col{{flex:1;min-width:240px;border:1px solid #ccc;border-radius:8px;padding:0 1rem}}
.delta{{color:#2e7d32}}.stale,.cautions{{background:#fff4e5;border-radius:8px;padding:.5rem 1rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ddd;padding:.25rem .6rem}}button{{padding:.6rem 1rem;margin-right:.5rem}}.ok{{background:#2e7d32;color:#fff}}</style></head>
<body><h1>เปรียบเทียบโมเดลก่อนอนุมัติ</h1>
<p>ผู้สมัคร <code>{_e(candidate_id)}</code> · เทียบกับ Champion <b>{cur}</b> · label: <b>{_e(r['label_mode'])}</b>
 · ที่มา: {_e(r['provenance'])} · ลูกค้าที่กันไว้ประเมิน: {_i(r['holdout_size'])} คน</p>
<p>ข้อมูลที่ใช้เทรน:<br>{files}<br>ช่วงเวลา {_e(str(dr[0]))} ถึง {_e(str(dr[1]))}</p>
<div class="cols">{_col('Champion (ปัจจุบัน)', r['champion'])}{_col('Challenger (เทรนใหม่)', r['challenger'], gate.get('growth_pct'))}</div>
<h3>ผลประเมินด่านอัตโนมัติ: {badge}</h3><p>{_e(gate['reason'])}</p>
<h3>รายร้าน</h3><table><tr><th>ร้าน</th><th>แถวข้อมูล</th><th>ลูกค้า</th><th>ลูกค้าที่กันไว้</th></tr>{store_rows}</table>
<h3>ข้อควรระวัง</h3><ul class="cautions">{caution_html}</ul>
{action}</body></html>"""
