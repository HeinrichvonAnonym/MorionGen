import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from motiogen import request_latency


class RequestLatencyTests(unittest.TestCase):
    def test_matched_ardy_windows_exclude_same_twenty_startup_requests(self):
        samples=[1000.]*20+list(range(30))
        result=request_latency.matched_ardy_steady_metrics(samples)
        self.assertEqual(result['matched_full_history_request_count'],30)
        self.assertEqual(result['matched_full_history_request_mean_seconds'],14.5)
        self.assertEqual(result['matched_full_history_request_p50_seconds'],14.5)
        self.assertAlmostEqual(result['matched_full_history_request_p95_seconds'],27.55)
        self.assertEqual(result['matched_full_history_request_rtf'],36.25)
        for invalid in [samples[:-1],samples+[1.],[-1.]+samples[1:],[float('inf')]+samples[1:]]:
            with self.assertRaises(ValueError):request_latency.matched_ardy_steady_metrics(invalid)

    def test_recorded_samples_preserve_startup_and_order(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(request_latency,'ROOT',Path(folder)):
            request_latency.recorded_requests.cache_clear()
            path=Path(folder)/'run';path.mkdir()
            (path/'worker_result.json').write_text(json.dumps({'timing':{'replan':{'samples_seconds':[2.,.1,.4]}}}))
            result=request_latency.recorded_requests('ardy','run')
            self.assertEqual(result['samples_seconds'],[2.,.1,.4])
            self.assertTrue(result['includes_startup_requests'])
            (path/'trace.json').write_text(json.dumps({'steps':[{'seconds':3.},{'seconds':.02}]}))
            self.assertEqual(request_latency.recorded_requests('closd','run')['samples_seconds'],[3.,.02])
            self.assertIn('excludes separately timed PHC',request_latency.recorded_requests('closd','run')['scope'])
            self.assertIsNone(request_latency.recorded_requests('primal','run'))
        request_latency.recorded_requests.cache_clear()

    def test_missing_or_negative_durations_not_filled(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(request_latency,'ROOT',Path(folder)):
            path=Path(folder)/'run';path.mkdir()
            for samples in [[],[-.1],[float('nan')]]:
                request_latency.recorded_requests.cache_clear()
                (path/'trace.json').write_text(json.dumps({'steps':[{'seconds':v} for v in samples]}))
                with self.assertRaises(ValueError):request_latency.recorded_requests('dart','run')
        request_latency.recorded_requests.cache_clear()

