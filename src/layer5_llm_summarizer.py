import re
import ollama

def _fmt_num(value, fmt=",.2f"):
    """Format number safely, return '-' if None."""
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value) if value is not None else "-"


def _format_payload(payload: dict) -> str:
    """แปลง payload dict ให้เป็น text ที่ LLM อ่านง่าย รองรับทั้ง run_pipeline และ csv_pipeline"""
    # --- Header ---
    meta = payload.get("metadata", {})
    planning_month = meta.get("planning_month") or payload.get("planning_month", "-")
    engine_version = meta.get("engine_version") or payload.get("campaign_name", "-")
    total_customers = meta.get("total_unique_customers", "-")
    total_transactions = meta.get("total_transactions", "-")

    lines = []
    lines.append("=== ข้อมูลสรุป Campaign ===")
    lines.append(f"เดือน: {planning_month}")
    lines.append(f"Engine: {engine_version}")
    if total_customers != "-":
        lines.append(f"ลูกค้าทั้งหมด: {total_customers} คน")
    if total_transactions != "-":
        lines.append(f"ธุรกรรมทั้งหมด: {total_transactions} รายการ")

    # Layer 4 broadcast summary (if available)
    bcast = payload.get("broadcast_summary", {})
    if bcast:
        total_bc  = bcast.get("total_broadcast", "-")
        total_sup = bcast.get("total_suppressed", "-")
        sup_rate  = bcast.get("suppression_rate_pct", "-")
        lines.append(f"⚡ Layer 4 Broadcast Summary: ส่งได้ {total_bc} คน | Suppressed {total_sup} คน ({sup_rate}% ถูก block โดย CFO guardrail)")
    lines.append("")

    # --- Segments ---
    for seg in payload.get("target_segments", []):
        ue = seg.get("unit_economics", {})
        audit = seg.get("promo_audit", {})
        ab = seg.get("ab_testing", {})

        net_profit = ue.get("net_profit_per_head")
        seg_profit = ue.get("total_segment_projected_profit") or ue.get("total_projected_profit")
        margin_status = audit.get("margin_health") or ue.get("financial_status", "-")
        promo_desc = seg.get("promo_desc") or seg.get("recommended_promo", "-")
        campaign = seg.get("campaign_strategy", "-")
        rationale = audit.get("rationale") or seg.get("rationale", "-")
        send_schedule = seg.get("send_schedule", "-")
        duration = seg.get("campaign_duration", "-")

        seg_name = seg.get('segment_name', '-')
        lines.append(f"--- [SEGMENT_NAME_EXACT: {seg_name}] (Cluster {seg.get('cluster_id', '-')}) ---")
        lines.append(f"⚠️ ใช้ชื่อกลุ่มนี้ตรงๆ ห้ามเปลี่ยน: {seg_name}")
        total_count    = seg.get('customer_count', '-')
        broadcast_count = seg.get('broadcast_count')
        suppressed_count = seg.get('suppressed_count')
        if broadcast_count is not None:
            lines.append(f"สัดส่วน: {seg.get('customer_share_percent', '-')} | จำนวนทั้งหมด: {total_count} คน")
            lines.append(f"⚡ Layer 4 Approved (ส่งได้): {broadcast_count} คน | Suppressed (ห้ามส่ง): {suppressed_count} คน ({seg.get('suppression_rate', 0)}% ถูก block โดย CFO guardrail)")
            l4_profit = seg.get('unit_economics', {}).get('l4_total_projected_profit')
            if l4_profit is not None:
                lines.append(f"⚡ L4 Projected Profit (เฉพาะคนที่ควรส่ง): ${_fmt_num(l4_profit)}")
        else:
            lines.append(f"สัดส่วน: {seg.get('customer_share_percent', '-')} | จำนวน: {total_count} คน")
        lines.append(f"RFM: ซื้อล่าสุด {seg.get('avg_recency_days', '-')} วัน | ความถี่ {seg.get('avg_frequency', '-')} ครั้ง | AOV ${_fmt_num(seg.get('avg_monetary'))}")
        if ab:
            lines.append(f"A/B Split: Treatment {ab.get('treatment_group_count')} คน / Control {ab.get('control_group_count')} คน")
        lines.append(f"Campaign: {campaign}")
        lines.append(f"ข้อเสนอ: {promo_desc}")
        if audit.get("promo_cost_per_head"):
            lines.append(f"Promo Cost/Head: {audit.get('promo_cost_per_head')}")
        lines.append(f"Net Profit/Head: ${_fmt_num(net_profit)}")
        lines.append(f"Margin Status: {margin_status}")
        lines.append(f"เหตุผล: {rationale}")
        if seg_profit is not None:
            lines.append(f"Projected Profit (กลุ่มนี้): ${_fmt_num(seg_profit)}")
        lines.append(f"กำหนดส่ง: {send_schedule} | ระยะเวลา: {duration}")

        hist = seg.get("historical_metrics", {})
        if hist:
            lines.append(f"[ข้อมูลจริงจากอดีต] ช่วงข้อมูล: {hist.get('data_period', '-')}")
            lines.append(f"  - Historical Retention Rate: {hist.get('retention_rate_pct')}%")
            lines.append(f"  - Historical Repeat Purchase Rate: {hist.get('repeat_purchase_rate_pct')}%")

        fcst = seg.get("post_campaign_forecast", {})
        if fcst:
            lines.append(f"[คาดการณ์หลังแคมเปญ] uplift สมมติฐาน: retention +{fcst.get('retention_uplift_assumption_pct')}%, repeat +{fcst.get('repeat_uplift_assumption_pct')}%")
            lines.append(f"  - Projected Retention Rate: {fcst.get('projected_retention_rate_pct')}%")
            lines.append(f"  - Projected Repeat Purchase Rate: {fcst.get('projected_repeat_purchase_rate_pct')}%")

        lines.append("")

    # --- Financial Summary ---
    summary = payload.get("campaign_financial_summary", {})
    lines.append("=== Financial Summary ===")
    if summary:
        lines.append(f"Total Projected Revenue: ${_fmt_num(summary.get('grand_total_projected_revenue'))}")
        lines.append(f"Baseline Revenue: ${_fmt_num(summary.get('baseline_projected_revenue'))}")
        lines.append(f"Revenue Lift: +{summary.get('revenue_lift_percent')}%")
        lines.append(f"Total Projected Profit: ${_fmt_num(summary.get('grand_total_projected_profit'))}")
        lines.append(f"Baseline Profit: ${_fmt_num(summary.get('baseline_projected_profit'))}")
        lines.append(f"Profit Lift: +{summary.get('profit_lift_percent')}%")
        lines.append(f"Validation Gate: {summary.get('validation_gate_status', '-')}")
    else:
        total_profit = payload.get('total_campaign_projected_profit', '-')
        lines.append(f"[REQUIRED: ต้องแสดงในรายงาน] Total Projected Profit (ทุกกลุ่มรวมกัน): {total_profit}")
        lines.append(f"Gate Status: {payload.get('gate_status', '-')}")

    return "\n".join(lines)


def _validate_output(output: str, payload: dict) -> list:
    """
    Cross-check LLM output against payload anchor values.
    Returns a list of warning strings for values not found in the output.
    """
    def num_variants(value):
        """Common string representations a number could appear as in text."""
        try:
            v = float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return set()
        return {
            f"{v:,.0f}", f"{v:.0f}", f"{v:,.2f}", f"{v:.2f}",
            str(int(round(v))),
        }

    def found(text, variants):
        return any(s in text for s in variants if s)

    warnings = []

    for seg in payload.get("target_segments", []):
        name = seg.get("segment_name", "")

        # Segment name must appear
        if name and name not in output:
            warnings.append(f"Segment name missing: '{name}'")

        # Customer count
        count = seg.get("customer_count")
        if count is not None and not found(output, num_variants(count)):
            warnings.append(f"Customer count {count} not found (segment: '{name}')")

        # Projected profit
        ue = seg.get("unit_economics", {})
        profit = ue.get("total_segment_projected_profit") or ue.get("total_projected_profit")
        if profit is not None and not found(output, num_variants(profit)):
            warnings.append(f"Segment profit {profit} not found (segment: '{name}')")

    # Total campaign profit
    summary = payload.get("campaign_financial_summary", {})
    total_profit = summary.get("grand_total_projected_profit")
    if total_profit is None:
        # csv_pipeline stores it as a formatted string e.g. "1,234.56 บาท"
        raw_str = payload.get("total_campaign_projected_profit", "")
        m = re.search(r"[\d,]+\.?\d*", str(raw_str))
        if m:
            total_profit = m.group().replace(",", "")
    if total_profit is not None and not found(output, num_variants(total_profit)):
        warnings.append(f"Total campaign profit not found in output")

    return warnings


def generate_campaign_action_plan(payload: dict) -> str:
    system_prompt = (
        "You are a senior CFO and CMO specializing in retail e-commerce strategy. "
        "Write a complete, professional Campaign Action Plan in Thai using the exact numbers provided — never invent or alter figures. "
        "CRITICAL RULES — violating these invalidates the entire report:\n"
        "1. SEGMENT NAMES: Each segment is labeled with [SEGMENT_NAME_EXACT: ...]. You MUST use that exact name verbatim. NEVER renumber, translate, shorten, or paraphrase segment names. If the name is '3. At-Risk / Churning Customers', write it exactly as '3. At-Risk / Churning Customers'.\n"
        "2. TOTAL PROFIT: Section 4 MUST include the total campaign projected profit. It is labeled [REQUIRED: ต้องแสดงในรายงาน]. Never omit this number.\n\n"
        "Use markdown formatting. Structure the report with exactly these 5 sections:\n\n"

        "## 1. สรุปภาพรวม\n"
        "ประกอบด้วย: เดือนที่วางแผน, จำนวนลูกค้าและธุรกรรม, จำนวนกลุ่มเป้าหมาย, สถานะ validation gate "
        "และ **Layer 4 Broadcast Summary** (⚡): จำนวนลูกค้าที่ผ่าน CFO guardrail ทั้งหมด vs ถูก suppress ทั้งหมด\n\n"

        "## 2. กลยุทธ์แคมเปญรายกลุ่ม\n"
        "สำหรับแต่ละ segment เขียนเป็น **bullet list กระชับ** (ห้ามใช้ subheadings ####) ดังนี้:\n"
        "- **ชื่อกลุ่ม**: (ใช้ชื่อตรงๆ จาก SEGMENT_NAME_EXACT)\n"
        "- **จำนวน**: X คน | Layer 4: ส่งได้ X คน / Suppressed X คน (X%)\n"
        "- **RFM**: ซื้อล่าสุด X วัน | ความถี่ X ครั้ง | AOV $X\n"
        "- **ข้อเสนอ**: ...\n"
        "- **Net Profit/Head**: $X | Margin: STATUS\n"
        "- **Projected Profit**: $X\n"
        "- **Rationale**: อธิบาย 1-2 ประโยค อ้างอิง RFM และ historical retention/repeat rate จริง\n\n"

        "## 3. แผนปฏิบัติการ (Execution Plan)\n"
        "ระบุ timeline week-by-week ตลอดระยะเวลาแคมเปญ เช่น สัปดาห์ที่ 1 เปิดตัว / สัปดาห์ที่ 2 ติดตามผล "
        "พร้อมบอกช่องทาง (Email / SMS) และกำหนดส่งที่แน่นอนจากข้อมูล\n\n"

        "## 4. ตารางสรุปการเงิน\n"
        "แสดง markdown table เปรียบเทียบ: กลุ่ม | จำนวนทั้งหมด | Broadcast (ส่งได้) | Suppressed (ห้ามส่ง) | AOV | Net Profit/Head | Projected Profit | Margin Status\n"
        "- หากไม่มีข้อมูล Layer 4 ให้แสดงคอลัมน์ Broadcast และ Suppressed เป็น '-'\n"
        "ตามด้วย total projected profit ทุกกลุ่มรวมกัน (ดูจากฟิลด์ [REQUIRED])\n\n"

        "## 5. การคาดการณ์และเหตุผลประกอบ\n"
        "แสดงผลที่คาดหวังพร้อม **เหตุผลประกอบ** ว่าทำไมถึงคาดการณ์ค่านั้น โดยอ้างอิงจาก:\n"
        "- Historical Retention Rate และ Repeat Purchase Rate จริงจากข้อมูล\n"
        "- Uplift assumption ที่ระบุ (+X% retention, +X% repeat) พร้อมอธิบายว่า assumption นี้มาจากอะไร\n"
        "- Projected Retention Rate และ Projected Repeat Purchase Rate หลังแคมเปญ\n"
        "- ความเสี่ยง (risk) ที่อาจทำให้ผลต่างจากที่คาด เช่น กลุ่ม At-Risk อาจไม่ตอบสนองโปร\n"
        "- KPIs ที่ต้องวัดจริง: Incremental Revenue Lift, Net Profit Lift\n"
        "- วิธีวัดผล A/B Testing และ next step ที่ควรทำ"
    )

    formatted_data = _format_payload(payload)
    user_prompt = f"เขียน Campaign Action Plan จากข้อมูลต่อไปนี้:\n\n{formatted_data}"

    try:
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            options={
                "temperature": 0.3,
                "repeat_penalty": 1.1,
                "num_predict": 7000,
            },
        )
        output = ""
        for chunk in response:
            token = chunk.message.content
            print(token, end="", flush=True)
            output += token
        print()  # newline after streaming ends
        warnings = _validate_output(output, payload)
        if warnings:
            warn_block = (
                "\n\n---\n"
                "⚠️ **[Validation Warnings]** ตัวเลขหรือชื่อกลุ่มต่อไปนี้ไม่พบใน output — "
                "LLM อาจแต่งขึ้นหรือปัดค่าผิด กรุณาตรวจสอบก่อนใช้งานจริง:\n"
            )
            warn_block += "\n".join(f"- {w}" for w in warnings)
            return output + warn_block
        return output
    except Exception as e:
        return f"❌ LLM Error: {str(e)}"
