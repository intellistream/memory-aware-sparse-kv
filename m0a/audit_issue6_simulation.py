"""Independently recompute result hashes, per-task time identities and saved event invariants."""
import csv
import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(sys.argv[1])
    hashes = json.loads((root/'artifacts-sha256.json').read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
    rows_checked = events_checked = records_checked = 0
    for path in sorted(root.glob('*/summary.json')):
        report = json.loads(path.read_text())
        assert not report['online_performance_validated']
        assert not report['autonomous_agent_validated']
        assert not report['numerical_attention_validated']
        assert not set(report['training_tasks']) & set(report['test_tasks'])
        for name, expected in report['code_sha256'].items():
            assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == expected
        with (path.parent/'per-task.csv').open() as f:
            for row in csv.DictReader(f):
                get = lambda k: float(row[k])
                assert abs(get('service_us')-get('wait_us')-get('lead_us')-get('attention_us')) < 1e-6
                assert abs(get('total_bytes_per_query')-get('prefetch_bytes_per_query')-get('sync_bytes_per_query')) < 1e-6
                assert 0 <= get('wasted_bytes_per_query') <= get('prefetch_bytes_per_query')+1e-6
                assert get('peak_reserved_bytes') <= get('capacity')*report['unit_bytes']
                assert 0 <= get('precision') <= 1 and 0 <= get('coverage') <= 1
                rows_checked += 1
        for ep in path.parent.glob('events-*.json'):
            events = json.loads(ep.read_text())
            transfers = events['transfers']
            for left, right in zip(transfers, transfers[1:]):
                assert left['end_us'] <= right['start_us']+1e-8
            by_query = {}
            for t in transfers:
                assert t['queued_us'] <= t['start_us'] <= t['end_us']
                by_query.setdefault((t['lane'], t['query']), []).append(t)
            for r in events['records']:
                assert r['prediction_ready_us'] <= r['prefetch_done_us'] <= r['attention_start_us']
                assert r['native_ready_us'] <= r['attention_start_us'] <= r['end_us']
                assert r['prefetch_units'] == r['useful_prefetch_units']+r['wasted_prefetch_units']
                transfer = by_query.get((r['lane'], r['query']), [])
                assert sum(t['units'] for t in transfer) == r['prefetch_units']+r['sync_recall_units']
                assert all(t['end_us'] <= r['attention_start_us']+1e-8 for t in transfer)
                records_checked += 1
            events_checked += 1
    result = dict(passed=True, artifact_hashes=len(hashes), per_task_rows=rows_checked,
                  event_files=events_checked, query_records=records_checked,
                  scope='simulation_output_consistency_only_not_hardware_or_model_validation')
    (root.parent/'simulation-audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
