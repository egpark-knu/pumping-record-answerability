"""Behavior tests for the D05 prescore statistics contract.

Synthetic rows only. This file does not read or write phase2 or phase3 results.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_statistics as st


def _row(**kwargs):
    base = dict(
        case_id="c0", realization=0, sid="confined_T50", storage_type="confined", T_m2_d=50,
        rho=0.25, N_transitions=6, Q_scale=1.0, SR=1.0, dataset="D03", experiment_group="D03",
        recency_label="d03_schedule", pair_id="P1_continue_vs_stop", track="raw", quantity="lead",
        lead_day=10, sign_determined=1, env_inf_m=0.2, env_sup_m=0.4, envelope_sign=1,
        W_m=0.2, W_status="ok", E_true_m=0.4, E_tool_m=0.3, E_reference_m=0.35,
    )
    base.update(kwargs)
    # Consistent synthetic outcomes; dedicated integration regressions test rejection.
    if kwargs.get("sign_determined")==0 and "env_inf_m" not in kwargs:
        base.update(env_inf_m=-.1,env_sup_m=.1,envelope_sign=0)
    if "W_m" in kwargs and "env_sup_m" not in kwargs:
        base["env_sup_m"]=base["env_inf_m"]+base["W_m"]
    if "W_m" not in kwargs:
        base["W_m"]=base["env_sup_m"]-base["env_inf_m"]
    return base


class PrespecificationTests(unittest.TestCase):
    def test_constants_are_locked_before_outcomes(self):
        self.assertEqual(st.SEED, 20261001)
        self.assertEqual(st.N_BOOTSTRAP, 999)
        self.assertEqual(st.RHO_OFFSET, 0.05)
        self.assertEqual(st.PROBABILITY_THRESHOLDS, (0.5, 0.9))
        self.assertIsNone(st.PRIMARY_PENALTY)
        self.assertEqual(st.REFERENCE_LAYER, "confined_T50")
        self.assertEqual(len(st.LAYER_ORDER), 6)
        self.assertEqual(st.EVAL_RHO, (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0))
        self.assertEqual(st.N_SIGNAL_BINS, 5)
        self.assertEqual(st.SEPARATION_ABS_COEF, 20.0)


class AlgebraTests(unittest.TestCase):
    def test_crossing_inverts_the_locked_linear_predictor(self):
        fit = st.ok_fit(
            {"intercept": 0.0, "log_signal_ratio": 2.0, "log_rho_plus_offset": 0.0},
            support=st.Support(-1.0, 2.0, 0.0, 4.0, (0.0, 1.0, 4.0)),
        )
        at_half = st.crossing_from_fit(fit, probability=0.5, rho=0.0, layer_id=None)
        at_nine = st.crossing_from_fit(fit, probability=0.9, rho=1.0, layer_id=None)
        self.assertAlmostEqual(at_half["signal_ratio"], 1.0, places=9)
        self.assertAlmostEqual(at_nine["signal_ratio"], math.exp(math.log(9) / 2), places=9)
        self.assertFalse(at_half["out_of_support"])
        self.assertEqual(at_half["reporting_status"], "finite_inside")

    def test_layer_coefficient_shifts_only_that_layer(self):
        fit = st.ok_fit(
            {
                "intercept": 0.0,
                "log_signal_ratio": 1.0,
                "log_rho_plus_offset": 0.0,
                "layer:leaky_T50": math.log(2),
            },
            support=st.Support(-2.0, 2.0, 0.0, 4.0, (1.0,)),
        )
        ref = st.crossing_from_fit(fit, probability=0.5, rho=1.0, layer_id="confined_T50")
        other = st.crossing_from_fit(fit, probability=0.5, rho=1.0, layer_id="leaky_T50")
        self.assertAlmostEqual(ref["signal_ratio"], 1.0, places=9)
        self.assertAlmostEqual(other["signal_ratio"], 0.5, places=9)

    def test_out_of_support_keeps_the_number(self):
        fit = st.ok_fit(
            {"intercept": -10.0, "log_signal_ratio": 1.0, "log_rho_plus_offset": 0.0},
            support=st.Support(0.0, 1.0, 0.25, 4.0, (0.25, 1.0, 4.0)),
        )
        found = st.crossing_from_fit(fit, probability=0.5, rho=0.0, layer_id=None)
        self.assertAlmostEqual(found["signal_ratio"], math.exp(10), places=6)
        self.assertTrue(found["sr_out_of_observed_support"])
        self.assertTrue(found["rho_not_observed"])
        self.assertEqual(found["reporting_status"], "extrapolated")
        self.assertIsNotNone(found["signal_ratio"])

    def test_zero_slope_does_not_invent_a_crossing(self):
        fit = st.ok_fit(
            {"intercept": 0.0, "log_signal_ratio": 0.0, "log_rho_plus_offset": 1.0},
            support=st.Support(-1.0, 1.0, 0.0, 1.0, (0.0, 1.0)),
        )
        found = st.crossing_from_fit(fit, probability=0.5, rho=1.0, layer_id=None)
        self.assertIsNone(found["signal_ratio"])
        self.assertEqual(found["status"], "zero_slope")
        self.assertEqual(found["reporting_status"], "unidentifiable")

    def test_failed_replicates_are_counted_and_not_filled_with_zeros(self):
        out = st.percentile_interval([1.0, 2.0, 3.0], n_not_identified=5)
        self.assertEqual(out["n_identified"], 3)
        self.assertEqual(out["n_not_identified"], 5)
        self.assertAlmostEqual(out["low"], float(np.percentile([1.0, 2.0, 3.0], 2.5)))
        self.assertGreater(out["low"], 0.5)


class FitTests(unittest.TestCase):
    def test_intercept_only_balanced_sample_is_zero(self):
        y = np.array([1.0, 0.0, 1.0, 0.0])
        X = np.ones((4, 1))
        fit = st.fit_logistic(y, X, column_names=("intercept",))
        self.assertEqual(fit["status"], "ok")
        self.assertAlmostEqual(fit["coefficients"]["intercept"], 0.0, places=6)
        self.assertAlmostEqual(fit["log_likelihood"], 4 * math.log(0.5), places=6)

    def test_separated_sample_is_not_identified_and_rows_remain_counted(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        X = np.column_stack([np.ones(4), np.array([-2.0, -1.0, 1.0, 2.0])])
        fit = st.fit_logistic(y, X, column_names=("intercept", "log_signal_ratio"))
        self.assertNotEqual(fit["status"], "ok")
        self.assertIsNone(fit["coefficients"])
        self.assertEqual(fit["n"], 4)

    def test_overlapping_sample_keeps_a_finite_positive_slope(self):
        # Both outcomes occur at each covariate value, with more failures on the left.
        x = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        X = np.column_stack([np.ones(8), x])
        fit = st.fit_logistic(y, X, column_names=("intercept", "log_signal_ratio"))
        self.assertEqual(fit["status"], "ok")
        self.assertGreater(fit["coefficients"]["log_signal_ratio"], 0.2)
        self.assertAlmostEqual(fit["coefficients"]["intercept"], 0.0, places=2)


class CohortTests(unittest.TestCase):
    def test_map_cohort_is_exact_n6_base_and_scaled_six_layers(self):
        rows = [
            _row(case_id="d03"),
            _row(case_id="b03", dataset="D04", experiment_group="B", recency_label="none", Q_scale=0.3, SR=0.3),
            _row(case_id="b30", dataset="D04", experiment_group="B", recency_label="none", Q_scale=3.0, SR=3.0),
            _row(case_id="n18", N_transitions=18),
            _row(case_id="recency", dataset="D04", experiment_group="A", recency_label="recent", N_transitions=2),
            _row(case_id="fixed", dataset="D04", experiment_group="C", sid="fixedc_leaky_T50", N_transitions=6),
            _row(case_id="new", dataset="D05", experiment_group="A", recency_label="none", rho=0.0, Q_scale=1.0),
        ]
        flags = [st.classify_row(st.adapt_row(r)) for r in rows]
        by_id = {r["case_id"]: f for r, f in zip(rows, flags)}
        self.assertTrue(by_id["d03"]["cohort_A"])
        self.assertTrue(by_id["b03"]["cohort_A"])
        self.assertTrue(by_id["b30"]["cohort_A"])
        self.assertTrue(by_id["new"]["cohort_A"])
        self.assertFalse(by_id["n18"]["cohort_A"])
        self.assertFalse(by_id["recency"]["cohort_A"])
        self.assertFalse(by_id["fixed"]["cohort_A"])
        self.assertTrue(all(not f["dropped"] for f in flags))

    def test_zero_signal_and_zero_truth_stay_in_the_inventory(self):
        zero_sr = st.adapt_row(_row(case_id="zsr", SR=0.0, sign_determined=1))
        zero_e = st.adapt_row(_row(case_id="ze", E_true_m=0.0, sign_determined=1, SR=1.0))
        zsr = st.classify_row(zero_sr)
        ze = st.classify_row(zero_e)
        self.assertTrue(zsr["cohort_A"])
        self.assertFalse(zsr["sign_likelihood"])
        self.assertEqual(zsr["sign_exclusion"], "nonpositive_signal")
        self.assertTrue(ze["sign_likelihood"])
        self.assertFalse(ze["magnitude_likelihood"])
        self.assertEqual(ze["magnitude_exclusion"], "zero_true_effect")
        self.assertFalse(zsr["dropped"] or ze["dropped"])

    def test_endpoint_touching_zero_is_not_determined_when_recomputed(self):
        row = st.adapt_row(_row(sign_determined="", envelope_sign="", env_inf_m=0.0, env_sup_m=0.4))
        self.assertEqual(row["sign_determined"], 0)
        self.assertEqual(row["includes_zero"], 1)
        positive = st.adapt_row(_row(sign_determined="", env_inf_m=0.1, env_sup_m=0.4, envelope_sign=""))
        self.assertEqual(positive["sign_determined"], 1)
        self.assertEqual(positive["envelope_sign"], 1)

    def test_phase2_header_aliases_build_a_layer_id(self):
        row = st.adapt_row({
            "source_phase": "phase2", "case_id": "r00_confined_T50_rho0.25_N6", "storage_type": "confined", "T_m2_d": "50.0",
            "rho": "0.25", "N_transitions": "6", "realization": "0", "pair_id": "P1_continue_vs_stop",
            "track": "raw", "quantity": "lead", "lead_day": "10", "W_m": "0.1", "W_status": "ok",
            "SR": "2.5", "E_true_m": "-0.4", "E_tool_m": "-0.2", "data_status": "official_analysis",
        })
        self.assertEqual(row["layer_id"], "confined_T50")
        self.assertEqual(row["Q_scale"], 1.0)
        self.assertEqual(row["lead_day"], 10)


class MagnitudeContourTests(unittest.TestCase):
    def test_width_contour_inverts_the_log_linear_mean(self):
        fit = st.ok_fit(
            {"intercept": -1.0, "log_signal_ratio": -1.0, "log_rho_plus_offset": 0.0},
            support=st.Support(-2.0, 2.0, 0.0, 4.0, (1.0,)),
        )
        found = st.crossing_from_fit(fit, probability=0.5, rho=1.0, layer_id=None)
        self.assertAlmostEqual(found["signal_ratio"], math.exp(-1.0), places=9)

    def test_log_linear_fit_recovers_a_known_slope(self):
        log_sr = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
        log_rho = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        y = -1.0 * log_sr
        X = np.column_stack([np.ones(6), log_sr, log_rho])
        fit = st.fit_log_linear(y, X, column_names=("intercept", "log_signal_ratio", "log_rho_plus_offset"))
        self.assertEqual(fit["status"], "ok")
        self.assertAlmostEqual(fit["coefficients"]["intercept"], 0.0, places=8)
        self.assertAlmostEqual(fit["coefficients"]["log_signal_ratio"], -1.0, places=8)
        self.assertAlmostEqual(fit["coefficients"]["log_rho_plus_offset"], 0.0, places=8)

    def test_zero_width_and_zero_truth_do_not_enter_the_log_contour(self):
        zero_w = st.classify_row(st.adapt_row(_row(W_m=0.0, E_true_m=0.4, SR=1.0)))
        zero_e = st.classify_row(st.adapt_row(_row(W_m=0.2, E_true_m=0.0, SR=1.0)))
        usable = st.classify_row(st.adapt_row(_row(W_m=1.0, E_true_m=0.4, SR=1.0)))
        self.assertFalse(zero_w["magnitude_likelihood"])
        self.assertEqual(zero_w["magnitude_exclusion"], "zero_width")
        self.assertFalse(zero_e["magnitude_likelihood"])
        self.assertEqual(zero_e["magnitude_exclusion"], "zero_true_effect")
        self.assertTrue(usable["magnitude_likelihood"])
        self.assertTrue(zero_w["sign_likelihood"] and zero_e["sign_likelihood"])


class SummaryTests(unittest.TestCase):
    def test_bins_are_five_log_equal_cuts_of_the_pooled_positive_signal(self):
        values = np.array([0.0, -1.0, 1.0, 10.0, 100.0, 100.0])
        bins = st.log_equal_bins(values)
        self.assertEqual(len(bins["edges"]), 6)
        self.assertAlmostEqual(bins["edges"][0], 1.0)
        self.assertAlmostEqual(bins["edges"][-1], 100.0)
        self.assertEqual(bins["n_nonpositive_signal"], 2)
        self.assertEqual(sum(bins["counts"]), 4)
        self.assertEqual(bins["n_bins"], 5)

    def test_contradiction_keeps_undetermined_truth_errors_separate(self):
        rows = [
            st.adapt_row(_row(case_id="bad", sign_determined=1, envelope_sign=1, E_tool_m=-0.2, E_reference_m=0.2, E_true_m=0.4, SR=1.0)),
            st.adapt_row(_row(case_id="zero_tool", sign_determined=1, envelope_sign=1, E_tool_m=0.0, E_reference_m=0.2, E_true_m=0.4, SR=1.2)),
            st.adapt_row(_row(case_id="open", sign_determined=0, envelope_sign=0, includes_zero=1, E_tool_m=-0.2, E_reference_m=-0.2, E_true_m=0.4, SR=0.2)),
            st.adapt_row(_row(case_id="quiet", sign_determined=0, envelope_sign=0, includes_zero=1, E_tool_m=0.2, E_reference_m=0.2, E_true_m=0.4, SR=0.3)),
        ]
        table = st.contradiction_table(rows, edges=st.log_equal_bins([r["SR"] for r in rows])["edges"])
        listed = [r.get("case_id") for block in table["by_bin_layer"] for r in block.get("rows",[])]
        self.assertEqual(table["tool_disagreement_including_zero"]["numerator"], 2)
        self.assertEqual(table["tool_contradiction"]["denominator"], 2)
        self.assertEqual(table["strict_opposite_sign"]["numerator"], 1)
        self.assertEqual(table["zero_tool_prediction"]["numerator"], 1)
        self.assertEqual(table["missing_tool"]["numerator"], 0)
        self.assertEqual(table["reference_agreement"]["numerator"], 2)
        self.assertEqual(table["reference_agreement"]["denominator"], 2)
        self.assertEqual(table["undetermined_truth_error"]["numerator"], 1)
        self.assertEqual(table["undetermined_truth_error"]["denominator"], 2)
        self.assertEqual(table["reference_undetermined_truth_disagreement"]["numerator"], 1)
        self.assertNotIn("bad", [r["case_id"] for r in table["undetermined_cases"]])
        self.assertNotIn("bad", listed)
        self.assertNotIn("zero_tool", listed)

    def test_calendar_denominator_includes_zero_signal_and_zero_truth(self):
        rows = [
            st.adapt_row(_row(case_id="ok", calendar_type="paddy", origin_month=5, dataset="D05", experiment_group="B")),
            st.adapt_row(_row(case_id="zsr", calendar_type="paddy", origin_month=5, dataset="D05", experiment_group="B", SR=0.0)),
            st.adapt_row(_row(case_id="ze", calendar_type="paddy", origin_month=5, dataset="D05", experiment_group="B", E_true_m=0.0, W_m=0.2)),
        ]
        rates = st.calendar_rates(rows)
        cell = rates["cells"][0]
        self.assertEqual(cell["denominator"], 3)
        self.assertEqual(cell["n_sign_determined"], 3)
        self.assertEqual(cell["n_zero_signal"], 1)
        self.assertEqual(cell["n_zero_true_effect"], 1)
        self.assertEqual(cell["n_relative_width_below_one"], 2)
        self.assertEqual(cell["n_relative_width_undefined"], 1)
        self.assertEqual(cell["relative_width_denominator"], 3)

    def test_width_equal_to_one_is_not_below_the_line(self):
        self.assertFalse(st.magnitude_resolved(1.0, 1.0))
        self.assertTrue(st.magnitude_resolved(0.999, 1.0))
        self.assertIsNone(st.magnitude_resolved(0.2, 0.0))

    def test_calendar_rate_interval_uses_the_supplied_realization_draw(self):
        rows = [
            st.adapt_row(_row(case_id="r0", realization=0, calendar_type="paddy", origin_month=5, sign_determined=1)),
            st.adapt_row(_row(case_id="r1", realization=1, calendar_type="paddy", origin_month=5, sign_determined=0, env_inf_m=-0.1, env_sup_m=0.1, envelope_sign=0)),
        ]
        draws = [{0: 1, 1: 0}, {0: 0, 1: 1}]
        cell = st.calendar_rates(rows, draws=draws)["cells"][0]
        self.assertEqual(cell["sign_rate"]["numerator"], 1)
        self.assertEqual(cell["sign_rate"]["denominator"], 2)
        self.assertEqual(cell["sign_rate"]["bootstrap"]["n_identified"], 2)
        self.assertAlmostEqual(cell["sign_rate"]["bootstrap"]["low"], float(np.percentile([0.0, 1.0], 2.5)))
        self.assertAlmostEqual(cell["sign_rate"]["bootstrap"]["high"], float(np.percentile([0.0, 1.0], 97.5)))

    def test_one_draw_is_shared_across_strata(self):
        draws = [{0: 2, 1: 0}]
        map_rows = [
            st.adapt_row(_row(case_id="a", realization=0, rho=0.25, Q_scale=1.0)),
            st.adapt_row(_row(case_id="b", realization=1, rho=4.0, Q_scale=3.0, pair_id="P2_current_vs_1p5x")),
        ]
        month_rows = [
            st.adapt_row(_row(case_id="m", realization=0, calendar_type="domestic", origin_month=1, dataset="D05")),
            st.adapt_row(_row(case_id="n", realization=1, calendar_type="domestic", origin_month=1, dataset="D05")),
        ]
        weighted_map = st.apply_draw(map_rows, draws[0])
        weighted_month = st.apply_draw(month_rows, draws[0])
        self.assertEqual([r["bootstrap_weight"] for r in weighted_map], [2.0, 0.0])
        self.assertEqual([r["bootstrap_weight"] for r in weighted_month], [2.0, 0.0])


class IntegrationTests(unittest.TestCase):
    def test_run_reports_every_requested_rho_without_dropping_rows(self):
        rows = []
        for realization in range(4):
            for layer in ("confined_T50", "leaky_T50"):
                for rho in (0.25, 1.0):
                    for scale, sr in ((0.3, 0.4), (1.0, 1.5)):
                        signal = sr * (1.2 if layer == "confined_T50" else 0.5)
                        determined = int(signal > 0.8)
                        rows.append(_row(
                            case_id=f"{layer}-{realization}-{rho}-{scale}",
                            realization=realization, sid=layer, storage_type=layer.split("_")[0],
                            rho=rho, Q_scale=scale, SR=signal, sign_determined=determined,
                            env_inf_m=(0.2 if determined else -0.1),
                            env_sup_m=((0.2 if determined else -0.1)+(0.2 if signal > 1 else 1.2)),
                            envelope_sign=(1 if determined else 0),
                            E_true_m=0.5, W_m=(0.2 if signal > 1 else 1.2),
                            E_tool_m=(0.4 if determined else -0.2), E_reference_m=0.4,
                            dataset="D04" if scale != 1 else "D03",
                            experiment_group="B" if scale != 1 else "D03",
                            recency_label="none" if scale != 1 else "d03_schedule",
                        ))
                        rows.append(_row(**{**rows[-1], "pair_id": "P2_current_vs_1p5x", "lead_day": 30, "case_id": rows[-1]["case_id"] + "-b"}))
        rows.append(_row(case_id="map-zero-sr", SR=0.0, sign_determined=0, env_inf_m=-0.1, env_sup_m=0.1, envelope_sign=0))
        rows.append(_row(
            case_id="calendar-zero", realization=0, dataset="D05", experiment_group="calendar",
            calendar_type="water_curtain", origin_month=1, SR=0.0, E_true_m=0.0, N_transitions=0,
            recency_label="none",
        ))
        result = st.run_statistics(rows, n_bootstrap=4, seed=20261001, official=False)
        self.assertEqual(result["inventory"]["n_input_rows"], len(rows))
        self.assertEqual(result["inventory"]["n_dropped"], 0)
        self.assertGreater(result["cohort_A"]["n_nonpositive_signal"], 0)
        self.assertIn("0.0", result["cohort_A"]["models"]["sign"]["10"]["P1_continue_vs_stop"]["no_layer"]["crossings"])
        zero = result["cohort_B"]["cells"]
        self.assertTrue(any(c["denominator"] >= 1 and c["n_zero_signal"] == 1 and c["n_zero_true_effect"] == 1 for c in zero))
        self.assertEqual(result["bootstrap"]["shared_across"], ["cohort_A", "cohort_B", "cohort_C", "pairs", "leads", "rho", "scales", "months"])
        self.assertEqual(result["prespecification"]["primary_penalty"], None)
        self.assertTrue(result["cohort_C"]["bins"]["pooled"])


if __name__ == "__main__":
    unittest.main()
