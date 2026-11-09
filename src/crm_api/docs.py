"""Thai documentation text shown in Swagger (/docs) and ReDoc (/redoc)."""
from __future__ import annotations

APP_TITLE = "CRM Campaign API"

APP_DESCRIPTION = """
API สำหรับให้ระบบ CRM ดึงผลของ **Campaign Engine** (การตัดสินใจจาก Layer 4 และตารางส่งรายเดือน)
ไปแสดงผล เช่น แผนแคมเปญ 1 เดือน ปฏิทินนัดส่ง โปรโมชัน และกลุ่มลูกค้า

**เกือบทั้งหมดอ่านอย่างเดียว** — เส้นแผน/กลุ่ม/ปฏิทินเพียงอ่านไฟล์ที่ pipeline เขียนไว้ ข้อมูลจึงสดเท่ากับการรัน pipeline ครั้งล่าสุด
ข้อยกเว้นคือกลุ่ม `/v1/runs` ที่ให้ระบบ CRM ส่งรายการขายเข้ามาให้รัน pipeline เอง (ทีละหนึ่งงาน) และดูผลแต่ละ Layer
งานที่ล้มเหลวไม่แตะแผนที่เผยแพร่ไว้แล้ว

### รอบ (wave) และการกระจายส่ง
- **รอบ (wave)** คือ *กลุ่มลูกค้าตามประเภทโปร*: รอบ 1 = ดึงลูกค้าช่วงเงียบ, รอบ 2 = เสนอขายช่วงยุ่ง, รอบ 3 = VIP ล่วงหน้า
  ไม่ใช่ "การส่ง 1 ครั้ง"
- แต่ละรอบถูก **กระจายส่งหลายวัน** (วันละกลุ่มย่อยคนละชุด) เพื่อไม่ให้ข้อความไปถึงทุกคนวันเดียวกัน
  ในปฏิทิน 1 รายการ = 1 กลุ่มย่อย จึงเห็นหลายรายการต่อรอบ
- ตามแผนปัจจุบัน **ลูกค้าแต่ละคนได้ข้อความครั้งเดียวต่อเดือน**

### กลุ่มลูกค้า (segmentation)
`/v1/segmentation*` คือกลุ่มลูกค้าที่ pipeline แบ่งจากประวัติการซื้อ **ไม่ผูกกับเดือนหรือแผนส่ง** — มีให้แม้ pipeline ตัดสินใจ
ไม่ส่งโปรให้ใครเลยก็ตาม จำนวนกลุ่มไม่ตายตัว (pipeline เลือกเอง) และเป็นคนละแนวคิดกับกลุ่ม RFM ของระบบภายนอก
ใช้เป็นป้ายเสริมได้ แต่ไม่ควรถือว่าเท่ากัน

### การยืนยันตัวตน
- ส่ง header `X-API-Key` ทุกครั้ง (ยกเว้น `/health`) — ใน Swagger กดปุ่ม **Authorize** แล้วใส่ key ครั้งเดียว
- ปฏิทิน `calendar.ics` รับ `?token=` ได้ด้วย เพราะแอปปฏิทินส่ง header ไม่ได้ token นี้ใช้ได้เฉพาะเส้นนั้น

### ข้อควรรู้
- `{month}` คือ **เดือนที่ส่งจริง** รูปแบบ `YYYY-MM` เช่น `2026-12`
- เวลาทั้งหมดเป็นเขตเวลา **Asia/Bangkok (+07:00)** และสกุลเงินคือ **THB**
- ลูกค้ากลุ่มควบคุม (holdout) และลูกค้าที่ระบบตัดสินใจไม่ส่ง จะไม่ปรากฏในรายการนัดส่งหรือปฏิทิน
- `503` แปลว่าไฟล์ข้อมูลกำลังถูกอัปเดตหรือไม่สอดคล้องกัน **ให้ลองใหม่อีกครั้ง** ระบบไม่ส่งแผนที่ไม่ครบ
- เส้นข้อมูลแผนส่ง header `ETag` กลับมา ส่งค่านั้นใน `If-None-Match` เพื่อเช็กว่าข้อมูลเปลี่ยนไหม (ได้ `304` ถ้าไม่เปลี่ยน) — ยกเว้น `calendar.ics` ที่ส่ง `ETag` แต่ยังไม่รองรับ `If-None-Match`
"""

TAGS = [
    {"name": "กลุ่มลูกค้า", "description": "กลุ่มลูกค้าที่ pipeline แบ่งไว้ (ไม่ผูกกับแผนรายเดือน) และลูกค้าแต่ละคนอยู่กลุ่มไหน"},
    {"name": "สถานะระบบ", "description": "ตรวจว่า API ทำงานอยู่ ไม่ต้องใช้ key"},
    {"name": "แผนแคมเปญ", "description": "สรุปแผนรายเดือน นัดส่งสำหรับปฏิทิน กลุ่มลูกค้า และโปรโมชัน"},
    {"name": "ลูกค้า", "description": "ข้อเสนอและเวลาส่งของลูกค้ารายคน"},
    {"name": "ปฏิทิน", "description": "ไฟล์ปฏิทิน .ics ให้แอปปฏิทิน subscribe"},
    {"name": "การรัน pipeline", "description": "ส่งรายการขายเข้ามาให้รัน pipeline ตามต้องการ ดูสถานะงาน และดูผลของแต่ละ Layer"},
    {"name": "ML แคมเปญ", "description": "รัน ML pipeline วางแผนแคมเปญ ตรวจสอบความเป็นไปได้ ดูประวัติ และบันทึกผลจริง"},
    {"name": "สุขภาพลูกค้า", "description": "วิเคราะห์สัดส่วน 10 กลุ่ม IBM RFM คะแนนสุขภาพฐานลูกค้า แนวโน้ม และ alert"},
    {"name": "เนื้อหา", "description": "สร้าง creative content สำหรับแคมเปญด้วย AI (ต้องการ ANTHROPIC_API_KEY)"},
    {"name": "กลุ่มเป้าหมาย", "description": "ส่งออก audience list สำหรับ Meta Ads / LINE OA"},
    {"name": "โมเดล", "description": "ตรวจสอบสถานะและ trigger การ retrain โมเดล ML"},
    {"name": "วันที่แนะนำ", "description": "คำนวณวันที่เหมาะสมที่สุดสำหรับส่งแคมเปญ โดยรวม buy-day affinity, payday window และ campaign overlap"},
]

ERRORS_TH = {
    401: "ไม่มี API key หรือ key ไม่ถูกต้อง — body เป็น `{\"error\": {\"code\", \"message\"}}`",
    404: "ไม่พบเดือน หรือไม่พบลูกค้าในแผนนั้น",
    422: "รูปแบบเดือน (`YYYY-MM`) หรือพารามิเตอร์ไม่ถูกต้อง",
    503: "ไฟล์ข้อมูลหายหรืออ่านไม่ได้ หรือไม่สอดคล้องกับไฟล์สรุป (กำลังอัปเดต?) — ลองใหม่ภายหลัง",
}

SEC_HEADER = "API key ของระบบ (ตั้งค่าที่ตัวแปรสภาพแวดล้อม `CRM_API_KEY`) ใช้กับทุกเส้นยกเว้น `/health`"
SEC_QUERY = ("Token สำหรับอ่านปฏิทินอย่างเดียว (ตั้งค่าที่ `CRM_API_CALENDAR_TOKEN`) "
             "ใช้ได้เฉพาะ `calendar.ics` เพราะแอปปฏิทินส่ง header ไม่ได้")

P_MONTH = "เดือนที่ส่งจริงของแผน รูปแบบ `YYYY-MM` เช่น `2026-12`"
P_FROM = "กรองเฉพาะนัดส่งตั้งแต่วันนี้เป็นต้นไป รูปแบบ `YYYY-MM-DD` (รวมวันนี้) เว้นว่างเพื่อไม่กรอง"
P_TO = "กรองเฉพาะนัดส่งจนถึงวันนี้ รูปแบบ `YYYY-MM-DD` (รวมวันนี้) ต้องไม่ก่อน `from`"
P_WAVE = "กรองตามรอบการส่ง: `1` = ดึงลูกค้าช่วงเงียบ (off-peak), `2` = เสนอขายช่วงยุ่ง (peak), `3` = VIP ล่วงหน้า"
P_EVENT_ID = "รหัสนัดส่ง (`id` ในรายการ `events`) ใช้ลิงก์ `customers_url` ที่ให้มาได้เลย ไม่ต้องประกอบเอง"
P_CURSOR = "เครื่องหมายต่อหน้า — ใส่ค่า `next_cursor` ของหน้าก่อนหน้า เว้นว่างเพื่อเริ่มจากหน้าแรก"
P_LIMIT = "จำนวนลูกค้าต่อหน้า (1–1000) ค่าตั้งต้น 100"
P_SEGMENT_ID = "กรองเฉพาะลูกค้าในกลุ่มนี้ (`id` ของกลุ่มจาก `/v1/segmentation`) เว้นว่างเพื่อเอาทุกกลุ่ม"
P_CUSTOMER = "รหัสลูกค้า (`customer_id`) ตามที่ปรากฏในแผน"

HEALTH = dict(
    tags=["สถานะระบบ"], summary="ตรวจสถานะระบบ",
    description=("ตรวจว่า API ทำงานอยู่ และแสดงรายชื่อเดือนที่มีแผนพร้อมใช้งาน "
                 "(เดือนที่มีทั้งไฟล์สรุปและตารางส่ง)\n\n**ไม่ต้องใช้ API key** เหมาะกับ health check ของระบบเฝ้าระวัง"),
    response_description="ระบบทำงานปกติ พร้อมรายชื่อเดือนที่มีแผน (`months_available`)",
)
MONTHS = dict(
    tags=["แผนแคมเปญ"], summary="รายชื่อเดือนที่มีแผน",
    description=("แสดงเดือนทั้งหมดที่ CRM เรียกดูแผนได้ เรียงจากเก่าไปใหม่ "
                 "เดือนจะปรากฏเมื่อมีทั้งไฟล์สรุป (`campaign_summary`) และตารางส่ง (`campaign_schedule`)"),
    response_description="รายชื่อเดือนรูปแบบ `YYYY-MM`",
)
PLAN = dict(
    tags=["แผนแคมเปญ"], summary="สรุปแผนแคมเปญรายเดือน",
    description=("ภาพรวมแผน 1 เดือน: จำนวนลูกค้าที่ถูกจัดส่ง จำนวนกลุ่มควบคุม (holdout) กำไรคาดการณ์รวมและเฉลี่ยต่อคน "
                 "จำนวนลูกค้าในแต่ละรอบ (wave) พร้อมสัปดาห์ที่ส่ง ชั่วโมง peak/off-peak ที่ตรวจพบ "
                 "และสถานะ validation gate\n\n"
                 "`weeks` คือสัปดาห์ที่รอบนั้นส่ง (ลูกค้าแต่ละคนถูกกำหนดไว้สัปดาห์เดียว ไม่ได้ส่งทั้งสองสัปดาห์) "
                 "และแต่ละรอบถูกกระจายส่งหลายวัน — ดูรายละเอียดรายนัดที่ `events`\n\n"
                 "ค่า `gate_status` และ `weeks` จะเป็น `null` ถ้าไฟล์กลุ่มลูกค้า (`segments`) ของเดือนนั้นยังไม่ถูกสร้าง"),
    response_description="สรุปแผนของเดือนที่ขอ (มี header `ETag`)",
)
EVENTS = dict(
    tags=["แผนแคมเปญ"], summary="รายการนัดส่งสำหรับแสดงในปฏิทิน",
    description=("นัดส่งของเดือนนั้น โดย 1 รายการ = 1 ช่วงเวลา (วันที่-เวลา) × 1 รอบ (wave) × 1 ช่วง (peak/off-peak) "
                 "บอกจำนวนลูกค้า โปรโมชันที่ส่ง ช่องทาง และกำไรคาดการณ์ของนัดนั้น\n\n"
                 "แต่ละนัดมี `customers_url` สำหรับดึง **รายชื่อผู้รับของนัดนั้น** ไปส่งเป็นชุด (batch)\n\n"
                 "แต่ละรอบถูกกระจายส่งหลายวัน ฟิลด์ `batch_index` / `batch_total` บอกว่านัดนี้เป็นกลุ่มย่อยที่เท่าไหร่จากกี่กลุ่ม "
                 "ของรอบนั้น (เรียงตามเวลา) และ `wave_customer_total` คือจำนวนลูกค้าทั้งรอบ — ลูกค้าแต่ละคนได้ข้อความครั้งเดียว\n\n"
                 "`id` ของแต่ละนัดสร้างจากวัน-เวลา รอบ และช่วง จึงคงที่เมื่อสร้างแผนซ้ำด้วยข้อมูลเดิม ใช้อ้างอิงใน CRM ได้ "
                 "กรองด้วย `from`, `to`, `wave` ได้"),
    response_description="รายการนัดส่งเรียงตามเวลา (มี header `ETag`)",
)
CALENDAR = dict(
    tags=["ปฏิทิน"], summary="ไฟล์ปฏิทิน (.ics) ของแผนรายเดือน",
    description=("ไฟล์ iCalendar ที่แอปปฏิทิน (Google Calendar, Apple Calendar ฯลฯ) นำเข้าหรือ subscribe ได้ "
                 "1 นัดส่งต่อ 1 เหตุการณ์ ยาว 30 นาที เขตเวลา Asia/Bangkok\n\n"
                 "ยืนยันตัวตนได้สองแบบ: header `X-API-Key` **หรือ** `?token=` (token สำหรับปฏิทินโดยเฉพาะ ใช้กับเส้นอื่นไม่ได้)"),
    response_description="ไฟล์ `text/calendar` (UTF-8)",
)
SEGMENTS = dict(
    tags=["แผนแคมเปญ"], summary="กลุ่มลูกค้า (Segment) ของแผน",
    description=("กลุ่มลูกค้าที่ pipeline แบ่งไว้ พร้อมขนาดกลุ่ม จำนวนที่ส่ง/ไม่ส่ง โปรที่แนะนำ "
                 "การตัดสินใจของ Layer 4 ต่อกลุ่ม และ unit economics\n\n"
                 "ถ้ายังไม่มีไฟล์กลุ่มลูกค้าของเดือนนั้น จะตอบ `segments: null` พร้อมเหตุผลใน `reason` (ไม่ใช่ error)"),
    response_description="รายการกลุ่มลูกค้า (มี header `ETag`)",
)
PROMOTIONS = dict(
    tags=["แผนแคมเปญ"], summary="โปรโมชันที่ใช้ในแผน",
    description=("โปรโมชันแต่ละแบบที่ถูกใช้ในเดือนนั้น พร้อมรอบ (wave) ช่องทาง จำนวนลูกค้า กำไรคาดการณ์ "
                 "และตัวอย่างข้อความที่ส่งจริง 1 ข้อความ"),
    response_description="รายการโปรโมชัน (มี header `ETag`)",
)
CUSTOMER = dict(
    tags=["ลูกค้า"], summary="ข้อเสนอของลูกค้ารายคน",
    description=("โปรที่ลูกค้าคนนี้จะได้รับ เวลาส่ง รอบ ช่องทาง และข้อความที่จะส่ง\n\n"
                 "ตอบ `404` แบบเดียวกันทั้งกรณี **ไม่รู้จักลูกค้า**, **เป็นกลุ่มควบคุม (holdout)** และ **ระบบตัดสินใจไม่ส่ง** "
                 "โดยตั้งใจ เพื่อไม่ให้ CRM ปฏิบัติต่อกลุ่มควบคุมต่างจากเดิมจนกระทบผลการวัด A/B"),
    response_description="ข้อเสนอและเวลาส่งของลูกค้า (มี header `ETag`)",
)

ERR_409 = "แผนถูกอัปเดตระหว่างที่อ่านหน้าถัดไป — เริ่มอ่านนัดนี้ใหม่จากหน้าแรก"
EVENT_CUSTOMERS = dict(
    tags=["ลูกค้า"], summary="รายชื่อผู้รับของนัดส่ง (สำหรับส่งเป็นชุด)",
    description=("รายชื่อลูกค้าทั้งหมดที่ต้องส่งในนัดนั้น พร้อมโปร ข้อความ และช่องทางของแต่ละคน "
                 "เรียงตาม `customer_id` และ **แบ่งหน้า**: อ่านต่อด้วย `next_cursor` จนกว่าจะเป็น `null`\n\n"
                 "ไม่มีลูกค้ากลุ่มควบคุม (holdout) และลูกค้าที่ระบบตัดสินใจไม่ส่ง\n\n"
                 "ถ้าแผนถูกอัปเดตระหว่างอ่าน จะได้ `409` ให้เริ่มนัดนี้ใหม่ (กันผู้รับปนกันสองเวอร์ชัน) "
                 "API นี้ไม่ได้ส่งข้อความเอง และไม่มีสถานะ \"ส่งแล้ว\" — CRM ต้องบันทึกเอง"),
    response_description="ผู้รับหนึ่งหน้า พร้อม `next_cursor` (มี header `ETag`)",
)

SEGMENTATION = dict(
    tags=["กลุ่มลูกค้า"], summary="กลุ่มลูกค้าของ pipeline (ภาพรวม)",
    description=("กลุ่มลูกค้าที่ pipeline แบ่งไว้ครั้งล่าสุด พร้อมจำนวนลูกค้าในแต่ละกลุ่มและจำนวนรวม และเวลาที่สร้างข้อมูล "
                 "ไม่ผูกกับเดือนหรือแผนส่ง จึงตอบได้แม้ไม่มีแผนรายเดือนเลย\n\n"
                 "ถ้ายังไม่เคยรัน pipeline จะได้ `404 segmentation_not_found`"),
    response_description="รายการกลุ่มและจำนวนลูกค้า (มี header `ETag`)",
)
SEGMENTATION_CUSTOMERS = dict(
    tags=["กลุ่มลูกค้า"], summary="ลูกค้าทุกคนพร้อมกลุ่ม (แบ่งหน้า)",
    description=("รหัสลูกค้าทุกคนพร้อมกลุ่มที่อยู่ เรียงตามรหัส แบ่งหน้าด้วย `next_cursor` (อ่านต่อจนเป็น `null`) "
                 "กรองเฉพาะกลุ่มได้ด้วย `segment_id` `total` คือจำนวนที่ตรงเงื่อนไขทั้งหมด\n\n"
                 "ข้อมูลมีแค่รหัสลูกค้ากับกลุ่ม ไม่มีชื่อ เบอร์ หรือตัวเลขการซื้อ และไม่บอกว่าใครอยู่กลุ่มควบคุม "
                 "ถ้า pipeline ถูกรันใหม่ระหว่างที่อ่านหน้าถัดไปจะได้ `409` ให้เริ่มอ่านใหม่จากหน้าแรก"),
    response_description="ลูกค้าหนึ่งหน้าพร้อม `next_cursor` (มี header `ETag`)",
)
SEGMENTATION_CUSTOMER = dict(
    tags=["กลุ่มลูกค้า"], summary="กลุ่มของลูกค้ารายคน",
    description="กลุ่มที่ลูกค้าคนนี้อยู่ ตอบ `404` ถ้าไม่รู้จักลูกค้า (ไม่อยู่ในข้อมูลที่ pipeline ใช้ล่าสุด)",
    response_description="รหัสลูกค้า กลุ่ม และชื่อกลุ่ม (มี header `ETag`)",
)
ERR_SEGMENTATION = {
    404: {"description": "ยังไม่มีข้อมูลกลุ่มลูกค้า (ยังไม่เคยรัน pipeline) หรือไม่พบลูกค้ารายนี้"},
    409: {"description": "ข้อมูลกลุ่มลูกค้าถูกอัปเดตระหว่างที่อ่านหน้าถัดไป — เริ่มอ่านใหม่จากหน้าแรก"},
}


P_RUN_ID = "รหัสงานที่ได้จาก `POST /v1/runs` (ขึ้นต้นด้วย `run_`)"

RUNS_START = dict(
    tags=["การรัน pipeline"], summary="ส่งรายการขายและสั่งรัน pipeline",
    description=("ส่งรายการขายเป็นข้อความ CSV (หัวคอลัมน์ตามที่ pipeline อ่าน: `customer_id`, `timestamp`, `transaction_id`, `product_name`, "
                 "`quantity`, `unit_price`, `total_price`; ใส่ `is_treated` และ `post_promo_visit` ได้ถ้ามีประวัติโปร) พร้อมเดือนที่ต้องการแผน "
                 "(`month`, `YYYY-MM`) และสัดส่วนกลุ่มควบคุม (`holdout_rate` 0–0.2)\n\n"
                 "ตอบ `202` ทันทีพร้อม `run_id` แล้วรันเบื้องหลัง ติดตามด้วย `GET /v1/runs/{run_id}` "
                 "รันได้ครั้งละหนึ่งงาน ถ้ามีงานค้างอยู่จะได้ `409` ไฟล์ใหญ่เกิน 20 MB ได้ `413`\n\n"
                 "ผลจะถูกเผยแพร่ (แทนไฟล์ในโฟลเดอร์ข้อมูล) เมื่องาน **สำเร็จเท่านั้น** งานที่ล้มเหลวไม่แตะแผนเดิม"),
    response_description="รับงานแล้ว: `run_id` และสถานะ",
)

RUNS_STATUS = dict(
    tags=["การรัน pipeline"], summary="สถานะของงานที่สั่งรัน",
    description=("สถานะ `queued` → `running` → `succeeded` หรือ `failed` พร้อมขั้นตอนที่ทำอยู่ (`layers`, `publishing`) เวลาเริ่ม/จบ "
                 "และ `label_mode` (`real` = ใช้ประวัติโปรจริง, `proxy` = ประมาณเอง) ถ้าล้มเหลวจะมีเฉพาะรหัสสาเหตุใน `error.code` "
                 "ไม่มีเส้นทางไฟล์หรือรหัสลูกค้า"),
    response_description="สถานะงาน",
)

RUNS_LAYERS = dict(
    tags=["การรัน pipeline"], summary="ผลของแต่ละ Layer จากงานล่าสุดที่สำเร็จ",
    description=("ผล Layer 1–4 ของงานล่าสุดที่สำเร็จ: กลุ่มลูกค้า (จำนวนกลุ่ม ขนาด ค่าเฉลี่ย), คู่สินค้าและข้อเสนอ, ชั่วโมงพีคและสต็อก, "
                 "และการตัดสินใจส่งโปรต่อ action พร้อมคำเตือนและบันทึกของ pipeline รูปแบบเดียวกับผลวิเคราะห์ของ DERA AI "
                 "ยังไม่มีงานที่สำเร็จจะได้ `404`"),
    response_description="ผลของแต่ละ Layer",
)

ERR_RUNS_START = {409: {"description": "มีงานรันอยู่แล้ว (`run_in_progress`) รอให้เสร็จก่อน"},
                  413: {"description": "ไฟล์ CSV ใหญ่เกินกำหนด (`csv_too_large`)"}}

# ---------------------------------------------------------------------------
# Phase 2 — ML Campaign, Segment Health, Content, Audience, Model endpoints
# ---------------------------------------------------------------------------

P_STORE_ID   = "รหัสร้านค้า (store identifier) ที่ใช้จัดเก็บข้อมูลและประวัติแคมเปญ"
P_PLAN_ID    = "รหัสแผนแคมเปญ (`plan_id`) ที่ได้จาก POST /campaign/plan"
P_LIMIT_H    = "จำนวนรายการต่อหน้า (1–100) ค่าตั้งต้น 20"
P_OFFSET     = "ข้ามกี่รายการ (สำหรับ pagination) ค่าตั้งต้น 0"
P_REQUEST_ID = "รหัสคำขอ retrain (`request_id`) ที่ได้จาก POST /model/retrain-request"

CAMPAIGN_PLAN = dict(
    tags=["ML แคมเปญ"], summary="สร้างแผนแคมเปญด้วย ML",
    description=(
        "รับไฟล์ POS (CSV base64) → รัน ML pipeline ครบ 4 Layer → คืนแผนแคมเปญพร้อม:\n\n"
        "- **สรุปงบ / กำไรคาดการณ์ / ROI**\n"
        "- **action_breakdown**: จำนวนลูกค้าและค่าใช้จ่ายแยกตาม action type\n"
        "- **segment_breakdown**: จำนวนลูกค้าที่ ML เลือกแยกตาม segment\n"
        "- **feasibility**: เป้าหมายทำได้ไหม และ max profit ที่ทำได้จริง\n\n"
        "mode `budget` → ต้องระบุ `budget` (THB) | mode `target_profit` → ต้องระบุ `target_profit` (THB)\n\n"
        "ไฟล์ POS จะถูกบันทึกและนำไปใช้กับ `/segment/health` ของร้านนั้นโดยอัตโนมัติ"
    ),
    response_description="แผนแคมเปญพร้อม plan_id สำหรับอ้างอิง",
)

CAMPAIGN_ANALYZE = dict(
    tags=["ML แคมเปญ"], summary="ตรวจสอบความเป็นไปได้ของแผน",
    description=(
        "ประเมินแผนที่ผู้ใช้ร่างเองผ่าน 4 ด้าน:\n\n"
        "1. **profitability** — กำไรสุทธิหลังหักค่าใช้จ่ายเป็นบวกไหม\n"
        "2. **overlap** — มีลูกค้าที่เพิ่งได้รับ promo ภายใน 21 วันไหม\n"
        "3. **segment_readiness** — segment ที่เลือกมีลูกค้าเพียงพอไหม\n"
        "4. **budget_efficiency** — allocate งบดีแล้วหรือสามารถ optimize ได้\n\n"
        "คืน `overall_verdict`: `recommended` / `proceed_with_caution` / `not_recommended`\n\n"
        "ไม่บันทึกแผน — ใช้สำหรับ sanity check ก่อน commit"
    ),
    response_description="ผลการตรวจสอบ 4 ด้านพร้อม overall_verdict",
)

CAMPAIGN_HISTORY = dict(
    tags=["ML แคมเปญ"], summary="ประวัติแผนแคมเปญของร้านค้า",
    description=(
        "ดึงรายการแผนที่สร้างผ่าน `/campaign/plan` ทั้งหมดของร้าน เรียงจากใหม่ไปเก่า\n\n"
        "แต่ละรายการมี `plan_id`, วันที่สร้าง, โหมด, จำนวนลูกค้าที่เลือก, กำไรคาดการณ์ และ ROI\n\n"
        "รองรับ pagination ด้วย `limit` และ `offset`"
    ),
    response_description="รายการแผนเรียงจากใหม่ไปเก่า",
)

CAMPAIGN_RECORD_OUTCOME = dict(
    tags=["ML แคมเปญ"], summary="บันทึกผลจริงของแคมเปญ",
    description=(
        "บันทึกรายได้จริง จำนวนลูกค้าที่ตอบสนอง และระยะเวลาวัดผล เพื่อ:\n\n"
        "- เปรียบเทียบกับประมาณการของ ML\n"
        "- ใช้เป็น signal ในการ calibrate โมเดลรอบถัดไป\n\n"
        "ผลจะถูกจัดเก็บและนำไปคำนวณ calibration stats (conversion rate จริง vs คาด) "
        "คืน `calibration_stats` ถ้ามีประวัติ outcome เพียงพอ"
    ),
    response_description="ยืนยันการบันทึกและ calibration stats (ถ้ามี)",
)

AUDIENCE_EXPORT = dict(
    tags=["กลุ่มเป้าหมาย"], summary="ส่งออก audience list สำหรับ Meta Ads / LINE OA",
    description=(
        "รับ `plan_id` → ดึงรายชื่อลูกค้าที่ ML เลือก → จับคู่กับข้อมูล CRM (ถ้าให้มา) → ส่งออก:\n\n"
        "- **Meta Custom Audience**: hashed email/phone สำหรับ Facebook/Instagram\n"
        "- **LINE OA Audience**: LINE userId สำหรับ Broadcast\n\n"
        "`crm_data` เป็น optional (base64 CSV) ถ้าไม่ให้มาจะคืนแค่ customer_id list"
    ),
    response_description="ผลการ export พร้อม export_id และสถิติการจับคู่",
)

CONTENT_GENERATE = dict(
    tags=["เนื้อหา"], summary="สร้าง content แคมเปญด้วย AI",
    description=(
        "สร้างข้อความ / คำบรรยายโปรโมชัน / caption สำหรับแคมเปญโดยใช้ Anthropic Claude\n\n"
        "ต้องการตัวแปรสภาพแวดล้อม `ANTHROPIC_API_KEY` — ถ้าไม่มีจะตอบ `503`\n\n"
        "`brief_type` รองรับ 2 แบบ:\n"
        "- `ml_driven` — ML กำหนด target segment และ action type ให้\n"
        "- `brand_driven` — ผู้ใช้กำหนด tone, offer, และเนื้อหาเอง\n\n"
        "`brand_kit` ใส่ชื่อร้าน สี น้ำเสียง และ emoji ที่แบรนด์ใช้ประจำ"
    ),
    response_description="เนื้อหาที่สร้างขึ้น พร้อม content_id",
    responses={503: {"description": "ไม่มี ANTHROPIC_API_KEY — ฟีเจอร์นี้ยังไม่เปิดใช้งาน"}},
)

CAMPAIGN_EXECUTE = dict(
    tags=["ML แคมเปญ"], summary="ส่งแคมเปญผ่าน LINE OA หรือ Meta Ads",
    description=(
        "รับ `plan_id` → ดึง audience จากแผน → ส่งผ่าน channel ที่เลือก:\n\n"
        "- **line_oa**: ส่ง LINE Multicast message ให้ user_ids ทุกคนในแผน\n"
        "- **meta**: รัน Meta Ads pipeline 6 ขั้นตอน "
        "(upload audience → lookalike → campaign → ad set → creative → ad)\n\n"
        "อัปเดต `campaign_history.jsonl` ด้วย `executed_at`, `channel`, `delivery_stats`\n\n"
        "คืน `{execution_id, status, delivery_summary}` ทันที\n\n"
        "ต้องการ env vars:\n"
        "- LINE: `LINE_CHANNEL_ACCESS_TOKEN`\n"
        "- Meta: `META_ACCESS_TOKEN`, `META_AD_ACCOUNT_ID`, `META_PAGE_ID`"
    ),
    response_description="ผลการส่ง execution_id และ delivery_summary",
    responses={
        503: {"description": "ไม่มี credentials ของช่องทางที่เลือก"},
    },
)

MODEL_RETRAIN_REQUEST = dict(
    tags=["โมเดล"], summary="ส่งคำขอ retrain โมเดล (queue ไว้รอ scheduler)",
    description=(
        "บันทึกคำขอ retrain ลงคิว — scheduler จะ pickup ในรอบกลางคืนหรือตามกำหนด\n\n"
        "ต่างจาก `POST /model/retrain` ที่รัน background job ทันที endpoint นี้ใช้เมื่อ:\n\n"
        "- DotDash ต้องการแจ้งว่าข้อมูลใหม่พร้อมแล้ว (เช่น POS 12 เดือนครบ)\n"
        "- ร้านสังเกตว่าผลลัพธ์คลาดเคลื่อนและต้องการ recalibrate\n\n"
        "``psm_mode`` กำหนดวิธี PSM: ``discount`` (มีคอลัมน์ส่วนลด) · "
        "``campaign_list`` (มีประวัติ campaign) · ``proxy`` (ใช้ RFM proxy) · "
        "``auto`` (ระบบเลือกให้)"
    ),
    response_description="request_id และสถานะ queued",
)

MODEL_RETRAIN_REQUEST_STATUS = dict(
    tags=["โมเดล"], summary="ตรวจสอบสถานะคำขอ retrain",
    description=(
        "คืนสถานะปัจจุบันของคำขอ retrain ที่ส่งผ่าน `POST /model/retrain-request`\n\n"
        "สถานะที่เป็นไปได้:\n"
        "- `queued` — รอ scheduler pickup\n"
        "- `processing` — กำลัง retrain\n"
        "- `completed` — retrain สำเร็จ\n"
        "- `failed` — retrain ล้มเหลว (ดู field `error`)"
    ),
    response_description="สถานะคำขอ retrain",
)

MODEL_RETRAIN = dict(
    tags=["โมเดล"], summary="สั่ง retrain โมเดล ML (async)",
    description=(
        "Trigger การ retrain PSM R-Learner uplift model ในพื้นหลัง\n\n"
        "คืน `job_id` ทันที — ตรวจสอบสถานะได้ที่ `GET /v1/model/status`\n\n"
        "mode `psm_mode` กำหนดวิธี identify treated group:\n"
        "- `discount` — ใช้แถวที่มี discount > 0\n"
        "- `campaign_list` — ใช้ customer list ที่เคยส่ง campaign\n"
        "- `proxy` — ประมาณจาก RFM pattern"
    ),
    response_description="job_id สำหรับตรวจสอบสถานะภายหลัง",
)

SEND_DATE_RECOMMEND = dict(
    tags=["วันที่แนะนำ"], summary="แนะนำวันที่ดีที่สุดสำหรับส่ง campaign",
    description=(
        "คำนวณคะแนนวันที่ทุกวันใน ``month`` จาก 3 สัญญาณ:\n\n"
        "- **buy-day affinity** — วันในสัปดาห์ / วันในเดือนที่ segment นี้ซื้อบ่อย (จาก POS ย้อนหลัง 90 วัน)\n"
        "- **payday window boost** — ช่วงเงินเดือนออก (25–สิ้นเดือน และ 1–5)\n"
        "- **campaign overlap penalty** — หักคะแนนถ้ามี campaign อื่นในช่วง cooldown\n\n"
        "ต้องอัปโหลด POS ผ่าน `/campaign/plan` ก่อน"
    ),
    response_description="วันที่แนะนำ, คะแนน, เหตุผล, และ alternatives",
)

SEGMENT_HEALTH = dict(
    tags=["สุขภาพลูกค้า"], summary="วิเคราะห์สุขภาพฐานลูกค้า (IBM 10 กลุ่ม)",
    description=(
        "รัน Layer 1 segmentation (RFM) บนไฟล์ POS ล่าสุดของร้าน → คืน 10 กลุ่ม IBM ครบเสมอ:\n\n"
        "- **count** / **trend** / **trend_pct**: จำนวน, ทิศทาง, % เปลี่ยนแปลงจาก snapshot ก่อน\n"
        "- **urgency_score**: 0–100 ความเร่งด่วนในการทำ campaign\n"
        "- **health_score**: สัดส่วน healthy / at_risk / churned และ label (ดี / ปานกลาง / ต้องดูแล)\n"
        "- **alerts**: กลุ่มที่โตผิดปกติ พร้อมระดับ warning/info\n"
        "- **recommendation**: เป้าหมาย campaign ที่แนะนำ (win_back / retain_loyal / maximize_profit)\n\n"
        "ต้องอัปโหลดไฟล์ผ่าน `/campaign/plan` ก่อนอย่างน้อย 1 ครั้ง\n\n"
        "snapshot จะถูกบันทึกทุกครั้งที่เรียก เพื่อใช้คำนวณ trend ในครั้งถัดไป"
    ),
    response_description="สุขภาพฐานลูกค้าปัจจุบัน พร้อม 10 กลุ่ม IBM",
)
