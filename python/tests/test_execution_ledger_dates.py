import datetime
import io
import json
import runpy
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import motiogen.config

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
REAL_DATETIME = datetime.datetime


class AuditClock(REAL_DATETIME):
    @classmethod
    def now(cls, tz=None):
        value = REAL_DATETIME(2026, 10, 25, 1, tzinfo=datetime.timezone.utc)
        return value.astimezone(tz) if tz is not None else value.replace(tzinfo=None)


class LedgerDatesTests(unittest.TestCase):
    def test_interruption_uses_recorded_bounds_not_recovery_wall_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=self.fixture(root);run=base/'20261025_example'
            manifest=json.loads((run/'manifest.json').read_text())
            manifest.update(status='interrupted_infrastructure',elapsed_seconds=None,
                            interruption_timing={'observed_elapsed_lower_seconds':3.})
            (run/'manifest.json').write_text(json.dumps(manifest))
            events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()][:1]
            events.append({'event':'queue_interruption_reconciled','wall_time':'2026-10-25T00:59:00+00:00',
                           'end_upper_bound':'2026-10-25T00:10:10+00:00'})
            (run/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            with patch.object(motiogen.config,'ROOT',root),patch('datetime.datetime',AuditClock),redirect_stdout(io.StringIO()):
                runpy.run_path(str(SCRIPTS/'native_execution_status.py'),run_name='__main__')
                runpy.run_path(str(SCRIPTS/'audit_native_execution.py'),run_name='__main__')
            status=json.loads((base/'20261024_native_execution/status.json').read_text())
            self.assertEqual(status['supervised_gpu_lease_hours']['1'],10/3600)
            self.assertEqual(status['observed_gpu_lease_hours_lower_bound']['1'],3/3600)
            audit=json.loads((base/'20261024_native_execution/audit.json').read_text())
            self.assertTrue(audit['passed_observed_checks'])
            interval=audit['intervals']['1'][0]
            self.assertEqual(interval['end']-interval['begin'],10)

    def fixture(self, root):
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        deadline = '2026-10-25T17:24:08Z'
        write(root / 'config/survey_revision.json', {
            'experiment_started_at': '2026-10-24T05:24:08Z',
            'experiment_deadline_at': deadline})
        base = root / 'develop_logs/test_runs'
        for date, stamp, device in [('20261023', '2026-10-23T10:00:00+00:00', 0),
                                    ('20261024', '2026-10-24T10:00:00+00:00', 0),
                                    ('20261025', '2026-10-25T00:10:00+00:00', 1),
                                    ('20261026', '2026-10-25T02:00:00+00:00', 1)]:
            run = base / (date + '_example')
            write(run / 'manifest.json', {'status': 'success', 'elapsed_seconds': 2,
                                         'budgets': {'gpu': {'index': device}}})
            begin = REAL_DATETIME.fromisoformat(stamp)
            events = [{'event': 'process_started', 'wall_time': stamp, 'pid': 2147483647},
                      {'event': 'run_finished', 'wall_time': (begin + datetime.timedelta(seconds=2)).isoformat()}]
            (run / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
            if date == '20261026':
                continue  # A receipt beyond this audit cutoff must not enter it.
            write(base / (date + '_matrix') / 'queue/state.json', {
                'pid': 2147483647, 'process_create_time': 0, 'status': 'finished',
                'heartbeat': stamp, 'deadline': deadline if date != '20261023' else '2026-10-23T17:24:08Z',
                'jobs': {'test': {'status': 'success'}}})
        return base

    def test_status_keeps_next_day_runs_and_same_deadline_queues(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = self.fixture(root)
            with patch.object(motiogen.config, 'ROOT', root), patch('datetime.datetime', AuditClock), redirect_stdout(io.StringIO()):
                runpy.run_path(str(SCRIPTS / 'native_execution_status.py'), run_name='__main__')
            result = json.loads((base / '20261024_native_execution/status.json').read_text())
            self.assertEqual(len(result['runs']), 2)
            self.assertEqual(len(result['queues']), 2)
            self.assertTrue(any('20261025_example' in r['run'] for r in result['runs']))
            self.assertEqual(result['supervised_gpu_lease_hours'], {'0': 2/3600, '1': 2/3600})
            self.assertEqual(result['deadline'], '2026-10-25T17:24:08+00:00')

    def test_gpu_audit_keeps_next_day_intervals(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = self.fixture(root)
            with patch.object(motiogen.config, 'ROOT', root), patch('datetime.datetime', AuditClock), redirect_stdout(io.StringIO()):
                runpy.run_path(str(SCRIPTS / 'audit_native_execution.py'), run_name='__main__')
            result = json.loads((base / '20261024_native_execution/audit.json').read_text())
            self.assertEqual(result['interval_counts'], {'0': 1, '1': 1})
            self.assertTrue(result['passed_observed_checks'])
            self.assertEqual(result['missing_terminal_intervals'], [])
            self.assertTrue('20261025_example' in result['intervals']['1'][0]['run'])
