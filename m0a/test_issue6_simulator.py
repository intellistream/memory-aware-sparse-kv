"""Analytical timing, contention, causality and accounting tests (CPU only)."""
from dataclasses import replace
import unittest

from .issue6_simulator import Timing, simulate, synthetic, predict
from .issue6_protocol import validate


def lane(selections):
    rows = synthetic(tasks=3, queries=len(selections), units=8, topk=2)
    rows = [r for r in rows if r['split'] == 'test']
    for row, selected in zip(rows, selections):
        row['selected_blocks'] = selected
    validate(rows)
    return rows


class SimulatorTests(unittest.TestCase):
    def test_transfer_units_and_launch_count(self):
        t = Timing(bandwidth_GBs=1, launch_us=2, batch_units=2)
        self.assertEqual(t.transfer_us(0, 1000), 0)
        self.assertEqual(t.transfer_us(3, 1000), 7)

    def test_cold_baseline_and_warm_hit_exact(self):
        t = Timing(bandwidth_GBs=1, launch_us=2, lead_us=10, attention_us=3)
        r, _, _ = simulate([lane([[0, 1], [0, 1]])], [[[], []]], capacity=2, budget=0,
                           unit_bytes=1000, timing=t)
        self.assertEqual([x['service_us'] for x in r], [17, 13])
        self.assertEqual([x['sync_recall_units'] for x in r], [2, 0])

    def test_perfect_prefetch_analytical_equation(self):
        for lead in (0, 2, 10):
            t = Timing(bandwidth_GBs=1, launch_us=2, lead_us=lead, prediction_us=1, attention_us=3)
            r, _, _ = simulate([lane([[0, 1]])], [[[0, 1]]], capacity=2, budget=2,
                               unit_bytes=1000, timing=t, strategy='last_selection')
            self.assertEqual(r[0]['service_us'], max(lead, 1+4)+3)
            self.assertEqual(r[0]['sync_recall_units'], 0)
            self.assertEqual(r[0]['useful_prefetch_units'], 2)

    def test_wrong_prefetch_pollution_and_conservation(self):
        t = Timing(bandwidth_GBs=1, launch_us=2, lead_us=0, prediction_us=0, attention_us=0)
        r, _, _ = simulate([lane([[0, 1], [0, 1]])], [[[], [2, 3]]], capacity=2, budget=2,
                           unit_bytes=1000, timing=t, strategy='last_selection')
        self.assertEqual(r[1]['service_us'], 8)
        self.assertEqual(r[1]['wasted_prefetch_units'], 2)
        self.assertEqual(r[1]['sync_recall_units'], 2)

    def test_fifo_shared_link_contention(self):
        t = Timing(bandwidth_GBs=1, launch_us=2, lead_us=0, attention_us=0)
        r, transfers, _ = simulate([lane([[0, 1]])]*2, [[[]]]*2, capacity=2, budget=0,
                                   unit_bytes=1000, timing=t)
        self.assertEqual(sorted(x['service_us'] for x in r), [4, 8])
        self.assertEqual(transfers[1]['start_us'], transfers[0]['end_us'])

    def test_late_signal_cannot_hide_transfer(self):
        t = Timing(bandwidth_GBs=1, launch_us=2, lead_us=10, signal_delay_us=20,
                   prediction_us=1, attention_us=0)
        r, _, _ = simulate([lane([[0, 1]])], [[[0, 1]]], capacity=2, budget=2,
                           unit_bytes=1000, timing=t, strategy='semantic_lsa')
        self.assertEqual(r[0]['wait_us'], 15)

    def test_echo_scores_not_early(self):
        t = Timing(bandwidth_GBs=1, launch_us=2, lead_us=10, prediction_us=1, attention_us=0)
        r, _, _ = simulate([lane([[0, 1]])], [[[0, 1]]], capacity=2, budget=2,
                           unit_bytes=1000, timing=t, strategy='echo_style')
        self.assertEqual(r[0]['prediction_ready_us'], 11)
        self.assertEqual(r[0]['wait_us'], 5)

    def test_zero_budget_equals_baseline(self):
        args = dict(capacity=2, budget=0, unit_bytes=1000)
        a, _, _ = simulate([lane([[0, 1]])], [[[]]], strategy='none', **args)
        b, _, _ = simulate([lane([[0, 1]])], [[[]]], strategy='global_frequency', **args)
        self.assertEqual(a[0]['service_us'], b[0]['service_us'])

    def test_reject_future_candidate_and_small_capacity(self):
        with self.assertRaisesRegex(ValueError, 'candidate'):
            simulate([lane([[0, 1]])], [[[8]]], capacity=2, budget=1, unit_bytes=1000)
        with self.assertRaisesRegex(ValueError, 'working set'):
            simulate([lane([[0, 1]])], [[[]]], capacity=1, budget=0, unit_bytes=1000)

    def test_c4_local_generation_never_host_transfer(self):
        rows = lane([[0, 1]] * 4)
        rows[-1]['selected_blocks'] = [1, 8]
        r, _, _ = simulate([rows], [[[]]*4], capacity=3, budget=0, unit_bytes=1024)
        self.assertEqual(sum(x['local_generated_units'] for x in r), 1)
        self.assertEqual(r[-1]['sync_recall_units'], 0)

    def test_historical_prediction_does_not_read_current_or_future_labels(self):
        a = lane([[0, 1], [2, 3], [4, 5]])
        b = lane([[0, 1], [6, 7], [1, 7]])
        pa = predict(a, 'global_frequency', 2, None, None)
        pb = predict(b, 'global_frequency', 2, None, None)
        self.assertEqual(pa[:2], pb[:2])
        self.assertEqual(pa[0], [])

    def test_generator_repeatability_and_all_scenarios_valid(self):
        for scenario in ('stable', 'shift', 'random', 'misleading'):
            a = synthetic(tasks=3, queries=8, units=64, topk=8, scenario=scenario)
            self.assertEqual(a, synthetic(tasks=3, queries=8, units=64, topk=8, scenario=scenario))
            validate(a)

    def test_invalid_timing(self):
        for t in (Timing(bandwidth_GBs=0), Timing(lead_us=-1), Timing(launch_us=float('nan'))):
            with self.assertRaises(ValueError):
                t.check()


if __name__ == '__main__':
    unittest.main()
