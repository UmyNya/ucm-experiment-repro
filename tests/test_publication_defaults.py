import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import common
from configuration import load
from inspect_sources import load_history


class PublicationDefaultsTest(unittest.TestCase):
    def test_run_does_not_copy_private_conversation_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'private').mkdir()
            (root / 'private' / 'conversation.txt').write_text('private fixture')
            config = load(ROOT / 'config/experiment.json')
            config['paths']['output_root'] = str(root / 'results')
            with patch.object(common, 'BUNDLE', root):
                run = common.new_run(config, 'fixture', 'privacy-test')
            self.assertTrue((run / 'configuration.json').is_file())
            self.assertFalse((run / 'private').exists())
            self.assertFalse(list(run.rglob('conversation.txt')))

    def test_history_map_accepts_relative_paths_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'map.json'
            source.write_text(json.dumps({'swe_small': 'experiments/small'}))
            self.assertEqual(load_history(source, root), {'swe_small': 'experiments/small'})
            for invalid in [{'../escape': 'valid'}, {'example': '../escape'},
                            {'example': str(root / 'absolute')}, {'example': '.'}]:
                source.write_text(json.dumps(invalid))
                with self.subTest(value=invalid), self.assertRaises(ValueError):
                    load_history(source, root)


if __name__ == '__main__':
    unittest.main()
