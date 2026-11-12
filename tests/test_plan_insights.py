"""plan_insights: rows by (segment, reward) with hours and days, the top-1 figure, model accuracy and the outcome summary."""
import pandas as pd
import pytest

from plan_insights import (CALIBRATION_BAND, hour_blocks, model_accuracy, outcome_summary, plan_rows, retrain_criteria,
                           top1_counterfactual, window_for, window_kind)
from retrain_trigger import PSI_RETRAIN, QINI_MIN_ACCEPTABLE


def _schedule():
    rows = []
    for i in range(10):   # S1 x A: peak, 10 people, profit 10..19, three dates
        rows.append(dict(customer_id=f"A{i}", segment_name="S1", recommended_action="A", action_key="a_key", action_group="win_back", wave=1,
                         send_window="Peak Window", send_datetime=f"2026-12-0{1 + i % 3} 10:00:00", expected_profit=10 + i))
    for i in range(4):    # S2 x B: off-peak, 4 people
        rows.append(dict(customer_id=f"B{i}", segment_name="S2", recommended_action="B", action_key=None, wave=2,
                         send_window="Off-Peak Window", send_datetime="2026-12-10 08:00:00", expected_profit=50))
    for i in range(2):    # S2 x C: both kinds in one row
        rows.append(dict(customer_id=f"C{i}", segment_name="S2", recommended_action="C", wave=3,
                         send_window="Peak Window" if i else "Off-Peak Window", send_datetime="2026-12-15 10:00:00", expected_profit=1))
    return pd.DataFrame(rows)


class TestHours:
    def test_blocks_are_contiguous_runs(self):
        assert hour_blocks([11, 12, 13, 17, 18]) == [(11, 14), (17, 19)]
        assert hour_blocks([]) == [] and hour_blocks([23]) == [(23, 24)] and hour_blocks([5, 5, 4]) == [(4, 6)]

    def test_window_is_the_block_from_the_earliest_hour_and_the_message_goes_the_hour_before(self):
        w = window_for("peak", [11, 12, 13, 17, 18], [9, 10])
        assert (w["start"], w["end"], w["notify_hour"], w["kind"]) == ("11:00", "14:00", 10, "peak")
        assert [b["start"] for b in w["all_blocks"]] == ["11:00", "17:00"]
        o = window_for("off_peak", [11], [9, 10])
        assert (o["start"], o["end"], o["notify_hour"]) == ("09:00", "11:00", 8)
        assert window_for("peak", [], [9]) is None
        assert window_for("peak", [23], [])["end"] == "23:59"   # the window runs to the end of the day

    def test_kind_follows_the_engines_own_rule(self):
        assert window_kind("Off-Peak Window") == "off_peak" and window_kind("Peak Window") == "peak" and window_kind(None) == "peak"


class TestRows:
    def test_one_row_per_segment_and_reward_best_first_with_hours_and_days(self):
        rows = plan_rows(_schedule(), None, [11, 12, 13], [9, 10])
        assert [(r["segment"], r["action"], r["rank"]) for r in rows] == [("S2", "B", 1), ("S1", "A", 2), ("S2", "C", 3)]
        a = next(r for r in rows if r["action"] == "A")
        assert (a["customers"], a["expected_profit"], a["profit_per_customer"], a["action_key"], a["action_group"]) == (10, 145.0, 14.5, "a_key", "win_back")
        assert (a["send_window"]["start"], a["send_window"]["end"], a["send_window"]["kind"]) == ("11:00", "14:00", "peak")
        assert a["recommended_dates"] == ["2026-12-01", "2026-12-02", "2026-12-03"] and (a["first_send_date"], a["last_send_date"]) == ("2026-12-01", "2026-12-03")
        b = next(r for r in rows if r["action"] == "B")
        assert b["send_window"]["kind"] == "off_peak" and b["action_key"] is None and b["waves"] == [2]

    def test_a_row_with_both_kinds_lists_both_and_leads_with_the_bigger(self):
        c = next(r for r in plan_rows(_schedule(), None, [11, 12], [9]) if r["action"] == "C")
        assert sorted(w["kind"] for w in c["send_windows"]) == ["off_peak", "peak"] and all(w["customers"] == 1 for w in c["send_windows"])

    def test_the_control_group_is_counted_per_row(self):
        hold = pd.DataFrame([dict(customer_id="H1", segment_name="S1", recommended_action="A"), dict(customer_id="H2", segment_name="S1", recommended_action="A"), dict(customer_id="H3", segment_name="S2", recommended_action="B")])
        rows = {(r["segment"], r["action"]): r["holdout_customers"] for r in plan_rows(_schedule(), hold, [11], [9])}
        assert rows == {("S1", "A"): 2, ("S2", "B"): 1, ("S2", "C"): 0}

    def test_empty_schedule_gives_nothing(self):
        assert plan_rows(pd.DataFrame(), None, [11], [9]) == [] and top1_counterfactual([]) is None


class TestTop1:
    def test_is_the_row_with_the_most_expected_profit_and_says_it_is_an_estimate(self):
        t = top1_counterfactual(plan_rows(_schedule(), None, [11], [9]))
        assert (t["segment"], t["action"], t["expected_profit"], t["plan_expected_profit"]) == ("S2", "B", 200.0, 347.0)
        assert t["share_of_plan_pct"] == pytest.approx(57.6, abs=0.1) and "ไม่ใช่ผลจริง" in t["note_th"] and t["rank_by"] == "expected_profit"


class TestAccuracy:
    def test_error_ratio_and_the_band(self):
        a = model_accuracy(1000, 900)
        assert (a["error_pct"], a["calibration_ratio"], a["within_band"], a["retrain_recommended"]) == (-10.0, 0.9, True, False)
        assert model_accuracy(1000, 300)["retrain_recommended"] is True and model_accuracy(1000, 1300)["retrain_recommended"] is True
        assert model_accuracy(1000, 800)["within_band"] is True and model_accuracy(1000, 1250)["within_band"] is True   # the edges are inside

    def test_no_prediction_or_no_actual_is_not_a_judgement(self):
        for p, v in [(0, 100), (None, 100), (100, None)]:
            a = model_accuracy(p, v)
            assert a["error_pct"] is None and a["within_band"] is None and a["retrain_recommended"] is False

    def test_an_interval_only_with_enough_customers_and_the_same_every_time(self):
        few = [(10.0, 9.0)] * 10
        assert model_accuracy(100, 90, few)["error_interval_pct"] is None
        many = [(10.0, 10.0 + (i % 5) - 2) for i in range(200)]
        i1 = model_accuracy(sum(p for p, _ in many), sum(a for _, a in many), many)["error_interval_pct"]
        assert i1 == model_accuracy(sum(p for p, _ in many), sum(a for _, a in many), many)["error_interval_pct"] and i1[0] <= 0 <= i1[1]

    def test_criteria_are_the_engines_own_thresholds_plus_the_band(self):
        c = {x["id"]: x for x in retrain_criteria()}
        assert c["feature_drift"]["threshold"] == PSI_RETRAIN and c["champion_qini"]["threshold"] == QINI_MIN_ACCEPTABLE and c["calibration"]["threshold"] == list(CALIBRATION_BAND)


class TestSummary:
    def test_a_good_result_keeps_going(self):
        r = outcome_summary(1000, 900, lift_pct=12.5)
        assert [a["code"] for a in r["next_actions"]] == ["KEEP_GOING"] and "คาดกำไร 1,000 บาท ผลจริงประมาณ 900 บาท (จริง/คาด 0.9)" in r["summary_th"] and "มากกว่ากลุ่มเทียบ 12.5%" in r["summary_th"]

    def test_a_miss_asks_for_a_retrain_and_a_negative_lift_is_said(self):
        r = outcome_summary(1000, 300, lift_pct=-4, window_complete=False)
        assert [a["code"] for a in r["next_actions"]] == ["WAIT_FOR_WINDOW", "REQUEST_RETRAIN", "PROMO_NO_LIFT"] and "น้อยกว่ากลุ่มเทียบ 4%" in r["summary_th"] and "ผลยังไม่ครบกำหนด" in r["summary_th"]

    def test_measurable_campaigns(self):
        none = outcome_summary(1000, 1000, lift_pct=5, groups=[{"measurable": False}, {"measurable": False}])
        assert [a["code"] for a in none["next_actions"]] == ["RAISE_HOLDOUT"]
        some = outcome_summary(1000, 1000, lift_pct=5, groups=[{"measurable": True}, {"measurable": False}])
        assert "SOME_NOT_MEASURABLE" in [a["code"] for a in some["next_actions"]] and "วัดผลเทียบกลุ่มควบคุมได้ 1 จาก 2 แคมเปญ" in some["summary_th"]

    def test_without_numbers_it_says_so_and_never_divides_by_zero(self):
        r = outcome_summary(0, None)
        assert "ยังประมาณผลจริงไม่ได้" in r["summary_th"] and r["model_accuracy"]["error_pct"] is None
