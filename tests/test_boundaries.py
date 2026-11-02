import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from boundaries import LIFECYCLE, drain_state


class DrainTest(unittest.TestCase):
    def setUp(self):
        self.row = {'pid': 1, 'metrics_collected': True, 'lifecycle': dict.fromkeys(LIFECYCLE, 0),
                    'pending_dump_task_status': [{'complete': True, 'sources_released': True}],
                    'counters': {'posix_h2s_bytes_total': 8192, 'posix_dump_write_bytes_total': 8192},
                    'histogram_counts': {'posix_dump_queue_wait_duration_ms': 2, 'posix_dump_task_duration_ms': 2}}

    def test_physical_completion_not_source_release_alone(self):
        self.row['pending_dump_task_status'][0]['complete'] = False
        self.assertFalse(drain_state(self.row)['physically_idle'])

    def test_equal_tasks_with_incomplete_bytes_is_not_drained(self):
        self.row['counters']['posix_dump_write_bytes_total'] = 4096
        self.assertFalse(drain_state(self.row)['physically_idle'])

    def test_early_jobs_cannot_be_ignored(self):
        self.row['lifecycle']['early_jobs'] = 1
        self.assertFalse(drain_state(self.row)['physically_idle'])

    def test_missing_evidence_fails_instead_of_assuming_zero(self):
        del self.row['lifecycle']['early_jobs']
        with self.assertRaises(ValueError):
            drain_state(self.row)

    def test_native_errors_retained_even_if_idle(self):
        self.row['counters']['posix_load_errors_total'] = 1
        self.assertTrue(drain_state(self.row)['errors'])
        self.assertTrue(drain_state(self.row)['physically_idle'])
