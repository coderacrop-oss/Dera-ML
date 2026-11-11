# ตารางชื่อ action ↔ `action_key` (สำหรับ DotDash และ Dera)

**สัญญา:** `action_key` คงที่หลัง release — ชื่อ action เปลี่ยนได้ แต่ key ที่ปล่อยไปแล้วจะไม่ถูกลบหรือเปลี่ยนชื่อ (มีเทสต์ `tests/test_action_key_contract.py` บังคับ) เพิ่ม key ใหม่ได้ โปรดจับคู่ด้วย `action_key` ไม่ใช่ชื่อ

- `{bundle}` ในชื่อคือชื่อเซตสินค้าที่ถูกแทนค่าตอนส่ง (ชื่อจริงที่ลูกค้าได้จึงต่างกันตามกลุ่ม แต่ key เดียวกัน)
- ชื่อที่ไม่ได้เปลี่ยนในรอบนี้ ชื่อเก่า = ชื่อใหม่
- ชื่อที่ยังมีคำว่า "X3" (`VIP Win-Back: {bundle} + แต้มพิเศษ X3`) ยังไม่ได้เปลี่ยนชื่อ รอตกลงชื่อใหม่ในรอบถัดไป key จะไม่เปลี่ยน
- แถวเก่าที่ไม่มี `action_key` ยังแปลงจากข้อความได้ (legacy) แต่จะมี warning

| ชื่อเก่า | action_key | ชื่อใหม่ |
|---|---|---|
| VIP Win-Back: กลับมาเป็น VIP อีกครั้ง | `st_cantlose_winback` | VIP Win-Back: กลับมาเป็น VIP อีกครั้ง |
| Welcome: ยินดีต้อนรับ กลับมาอีกนะ! | `st_newcust_welcome` | Welcome: ยินดีต้อนรับ กลับมาอีกนะ! |
| VIP Reward: แจกแต้ม X2 | `st_vip_reward_offpeak` | VIP Reward: แจกแต้มพิเศษ 100 แต้ม |
| VIP Early Access | `st_vip_early_access_peak` | VIP Early Access |
| Loyalty Nurture: แต้มสะสม | `st_loyalty_nurture` | Loyalty Nurture: แต้มสะสม |
| Loyalty Reward: {bundle} + แต้มพิเศษ | `loyal_vip_bundle` | Loyalty Reward: {bundle} + แต้มพิเศษ |
| VIP Reward: แจกแต้ม X2 | `loyal_vip_reward` | VIP Reward: แจกแต้มพิเศษ 100 แต้ม |
| Loyal Offer: {bundle} | `loyal_standard` | Loyal Offer: {bundle} |
| Habit Builder: สะสมแต้ม เพิ่มความถี่ | `potloyal_habit_builder` | Habit Builder: สะสมแต้ม เพิ่มความถี่ |
| Loyalty Builder Off-Peak: แต้มพิเศษช่วงเงียบ | `potloyal_offpeak_vip` | Loyalty Builder Off-Peak: แต้มพิเศษช่วงเงียบ |
| Re-engage: {bundle} พิเศษสำหรับคุณ | `promising_reengage` | Re-engage: {bundle} พิเศษสำหรับคุณ |
| Re-engage Off-Peak: ส่วนลดพิเศษกลับมาหาเรา | `promising_offpeak` | Re-engage Off-Peak: ส่วนลดพิเศษกลับมาหาเรา |
| เราเป็นห่วง: {bundle} + ส่วนลดพิเศษ | `needattn_bundle` | เราเป็นห่วง: {bundle} + ส่วนลดพิเศษ |
| Re-engage: ส่วนลดพิเศษช่วงเงียบสำหรับคุณ | `needattn_offpeak_traffic` | Re-engage: ส่วนลดพิเศษช่วงเงียบสำหรับคุณ |
| เราคิดถึงคุณ: {bundle} ก่อนที่จะไม่ได้เจอกัน | `aboutsleep_urgent` | เราคิดถึงคุณ: {bundle} ก่อนที่จะไม่ได้เจอกัน |
| Last Chance: ส่วนลดพิเศษ กลับมาหาเราเลย | `aboutsleep_traffic_driver` | Last Chance: ส่วนลดพิเศษ กลับมาหาเราเลย |
| Welcome: สั่งครั้งที่ 2 รับส่วนลด 10% | `newcust_welcome_peak` | Welcome: สั่งครั้งที่ 2 รับส่วนลด 10% |
| Welcome Off-Peak: กลับมาช้อปรับส่วนลดพิเศษ | `newcust_welcome_offpeak` | Welcome Off-Peak: กลับมาช้อปรับส่วนลดพิเศษ |
| Early Win-Back: {bundle} ก่อนที่จะไม่กลับมา | `churnrisk_rescue_bundle` | Early Win-Back: {bundle} ก่อนที่จะไม่กลับมา |
| Churn Rescue: ส่วนลดพิเศษช่วงเงียบ เราคิดถึงคุณ | `churnrisk_offpeak_traffic` | Churn Rescue: ส่วนลดพิเศษช่วงเงียบ เราคิดถึงคุณ |
| VIP Win-Back: {bundle} + แต้มพิเศษ X3 | `cantlose_vip_winback` | VIP Win-Back: {bundle} + แต้มพิเศษ X3 |
| VIP Win-Back: กลับมาช้อปรับส่วนลดพิเศษ VIP | `cantlose_aggressive_offpeak` | VIP Win-Back: กลับมาช้อปรับส่วนลดพิเศษ VIP |
| VIP Bundle: {bundle} | `champ_vip_bundle` | VIP Bundle: {bundle} |
| VIP Early Access | `champ_early_access_peak` | VIP Early Access |
| VIP Early Access | `champ_early_access_offpeak` | VIP Early Access |
| VIP Reward: แจกแต้ม X2 | `champ_vip_reward` | VIP Reward: แจกแต้มพิเศษ 100 แต้ม |
| Discount 15%: ซื้อ {bundle} | `atrisk_discount_bundle` | Discount 15%: ซื้อ {bundle} |
| Standard Retention | `atrisk_standard_offpeak` | Standard Retention |
| Standard Retention | `atrisk_standard_peak` | Standard Retention |
| Off-Peak Traffic Driver: ส่วนลดพิเศษช่วงเงียบ | `atrisk_offpeak_traffic` | Off-Peak Traffic Driver: ส่วนลดพิเศษช่วงเงียบ |
| กลับมาช้อป: {bundle} ลด 20% | `hiber_reactivation_deep` | กลับมาช้อป: {bundle} ลด 20% |
| Reactivation: {bundle} | `hiber_reactivation_traffic` | Reactivation: {bundle} |
| Standard Retention (Festival Guard) | `mid_festival_guard` | Standard Retention (Festival Guard) |
| Off-Peak Traffic Driver: ส่วนลดพิเศษช่วงเงียบ | `mid_offpeak_traffic` | Off-Peak Traffic Driver: ส่วนลดพิเศษช่วงเงียบ |
| Off-Peak VIP Points: ดึงลูกค้าช่วงเงียบ (แต้มพิเศษ) | `mid_offpeak_vip_points` | Off-Peak VIP Points: ดึงลูกค้าช่วงเงียบ (แต้มพิเศษ) |
| VIP Reward: แจกแต้ม X2 | `mid_vip_reward_peak` | VIP Reward: แจกแต้มพิเศษ 80 แต้ม |
| VIP Reward: แจกแต้ม X2 | `mid_vip_reward_offpeak` | VIP Reward: แจกแต้มพิเศษ 100 แต้ม |
| VIP Early Access | `mid_vip_early_access_peak` | VIP Early Access |
| VIP Early Access | `mid_vip_early_access_offpeak` | VIP Early Access |
| Bundle Upsell: {bundle} | `mid_bundle_upsell_peak` | Bundle Upsell: {bundle} |
| Bundle Upsell: {bundle} | `mid_bundle_upsell_offpeak` | Bundle Upsell: {bundle} |
| Standard Retention | `mid_standard_peak` | Standard Retention |
| Standard Retention | `mid_standard_offpeak` | Standard Retention |
| Milestone Reward: สั่งครบ {basket:.0f} บาท รับ {pts} แต้ม | `low_milestone` | Milestone Reward: สั่งครบ {basket:.0f} บาท รับ {pts} แต้ม |
| Welcome: สั่งครั้งที่ 2 รับส่วนลด 10% | `low_welcome` | Welcome: สั่งครั้งที่ 2 รับส่วนลด 10% |
| เราคิดถึงคุณ: กลับมาหาเราก่อนที่จะไม่ได้เจอกัน | `low_abouttosleep_miss_you` | เราคิดถึงคุณ: กลับมาหาเราก่อนที่จะไม่ได้เจอกัน |
| Reactivation Nudge: กลับมาช้อปรับส่วนลดพิเศษ | `low_hibernating_nudge` | Reactivation Nudge: กลับมาช้อปรับส่วนลดพิเศษ |
| Early Win-Back: กลับมาหาเรา รับส่วนลดพิเศษ | `low_churn_winback` | Early Win-Back: กลับมาหาเรา รับส่วนลดพิเศษ |
| Bundle Upsell: เพิ่ม {bundle} | `low_bundle_upsell` | Bundle Upsell: เพิ่ม {bundle} |

## ของรางวัลที่แต่ละ key ส่งให้ DotDash (`dotdash_offer`)

| action_key | ของรางวัล | หมดอายุ (วัน) |
|---|---|---|
| `st_cantlose_winback` | คูปอง 15% | 14 |
| `st_newcust_welcome` | คูปอง 10% | 30 |
| `st_vip_reward_offpeak` | 100 แต้ม | 30 |
| `st_vip_early_access_peak` | 50 แต้ม | 30 |
| `st_loyalty_nurture` | 50 แต้ม | 30 |
| `loyal_vip_bundle` | 80 แต้ม | 30 |
| `loyal_vip_reward` | 100 แต้ม | 30 |
| `loyal_standard` | คูปอง 5% | 30 |
| `potloyal_habit_builder` | 60 แต้ม | 30 |
| `potloyal_offpeak_vip` | 60 แต้ม | 30 |
| `promising_reengage` | คูปอง 5% | 30 |
| `promising_offpeak` | คูปอง 15% | 14 |
| `needattn_bundle` | คูปอง 5% | 30 |
| `needattn_offpeak_traffic` | คูปอง 22% | 14 |
| `aboutsleep_urgent` | คูปอง 15% | 14 |
| `aboutsleep_traffic_driver` | คูปอง 22% | 7 |
| `newcust_welcome_peak` | คูปอง 10% | 30 |
| `newcust_welcome_offpeak` | คูปอง 10% | 30 |
| `churnrisk_rescue_bundle` | คูปอง 12% | 14 |
| `churnrisk_offpeak_traffic` | คูปอง 22% | 14 |
| `cantlose_vip_winback` | คูปอง 15% | 14 |
| `cantlose_aggressive_offpeak` | คูปอง 22% | 7 |
| `champ_vip_bundle` | 150 แต้ม | 30 |
| `champ_early_access_peak` | 80 แต้ม | 30 |
| `champ_early_access_offpeak` | 100 แต้ม | 30 |
| `champ_vip_reward` | 100 แต้ม | 30 |
| `atrisk_discount_bundle` | คูปอง 15% | 14 |
| `atrisk_standard_offpeak` | คูปอง 15% | 14 |
| `atrisk_standard_peak` | คูปอง 5% | 30 |
| `atrisk_offpeak_traffic` | คูปอง 22% | 14 |
| `hiber_reactivation_deep` | คูปอง 20% | 14 |
| `hiber_reactivation_traffic` | คูปอง 22% | 14 |
| `mid_festival_guard` | คูปอง 5% | 30 |
| `mid_offpeak_traffic` | คูปอง 22% | 14 |
| `mid_offpeak_vip_points` | 60 แต้ม | 30 |
| `mid_vip_reward_peak` | 80 แต้ม | 30 |
| `mid_vip_reward_offpeak` | 100 แต้ม | 30 |
| `mid_vip_early_access_peak` | 80 แต้ม | 30 |
| `mid_vip_early_access_offpeak` | 100 แต้ม | 30 |
| `mid_bundle_upsell_peak` | คูปอง 8% | 30 |
| `mid_bundle_upsell_offpeak` | คูปอง 15% | 14 |
| `mid_standard_peak` | คูปอง 5% | 30 |
| `mid_standard_offpeak` | คูปอง 15% | 14 |
| `low_milestone` | 50 แต้ม | 30 |
| `low_welcome` | คูปอง 10% | 30 |
| `low_abouttosleep_miss_you` | คูปอง 5% | 14 |
| `low_hibernating_nudge` | คูปอง 3% | 14 |
| `low_churn_winback` | คูปอง 12% | 14 |
| `low_bundle_upsell` | คูปอง 3% | 30 |

## หมวดของ action (`action_group`)

ใช้แยกประเภท action แทนการจับ regex จากชื่อ (ส่งมาใน `dotdash_offer.action_group` และคอลัมน์ `action_group` ของ CSV) ค่าที่เป็นไปได้: `win_back`, `retain`, `loyalty_reward`, `onboarding`, `upsell`, `other` — เพิ่มค่าใหม่ได้ ค่าเดิมไม่เปลี่ยนความหมาย; `operator_promo` (โปรของร้าน) อยู่หมวด `other`; ไม่มีโปร = ว่าง

> การจัดหมวดด้านล่างเป็นข้อเสนอของทีม ML จากชื่อ/ชั้นลูกค้าของแต่ละ action ขอให้ Dera ตรวจ

- **win_back**: `st_cantlose_winback`, `churnrisk_rescue_bundle`, `churnrisk_offpeak_traffic`, `cantlose_vip_winback`, `cantlose_aggressive_offpeak`, `low_churn_winback`, `hiber_reactivation_deep`, `hiber_reactivation_traffic`, `low_hibernating_nudge`, `aboutsleep_urgent`, `aboutsleep_traffic_driver`, `low_abouttosleep_miss_you`
- **retain**: `loyal_standard`, `promising_reengage`, `promising_offpeak`, `needattn_bundle`, `needattn_offpeak_traffic`, `atrisk_discount_bundle`, `atrisk_standard_offpeak`, `atrisk_standard_peak`, `atrisk_offpeak_traffic`, `mid_festival_guard`, `mid_offpeak_traffic`, `mid_standard_peak`, `mid_standard_offpeak`
- **loyalty_reward**: `st_vip_reward_offpeak`, `st_vip_early_access_peak`, `st_loyalty_nurture`, `loyal_vip_bundle`, `loyal_vip_reward`, `potloyal_habit_builder`, `potloyal_offpeak_vip`, `champ_vip_bundle`, `champ_early_access_peak`, `champ_early_access_offpeak`, `champ_vip_reward`, `mid_offpeak_vip_points`, `mid_vip_reward_peak`, `mid_vip_reward_offpeak`, `mid_vip_early_access_peak`, `mid_vip_early_access_offpeak`, `low_milestone`
- **onboarding**: `st_newcust_welcome`, `newcust_welcome_peak`, `newcust_welcome_offpeak`, `low_welcome`
- **upsell**: `mid_bundle_upsell_peak`, `mid_bundle_upsell_offpeak`, `low_bundle_upsell`
