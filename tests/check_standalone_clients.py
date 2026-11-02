#!/usr/bin/env python3
"""CPU-only real-client integration: exact payloads, warmup, metrics, external service survival."""
import argparse
import array
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from common import http, read_json, save_json, new_run
from configuration import load
from processes import OwnedProcess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--client', choices=['aisbench', 'swe'], required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--reference-inputs', type=Path, required=True)
    a = p.parse_args()
    c = load(ROOT/'config/experiment.json', [f'paths.{a.client}_python={a.python}'])
    profile = 'aisbench_n40_performance' if a.client == 'aisbench' else 'swe_w24_n192_performance'
    evidence = new_run(c, profile, 'standalone-mock-validation')
    expected = {}
    if a.client == 'aisbench':
        manifest = read_json(a.reference_inputs/'manifest.json')
        for kind in ['formal', 'warmup']:
            texts = [json.loads(line)['question'] for line in (a.reference_inputs/(kind+'.jsonl')).read_text().splitlines()]
            for text, tokens in zip(texts, manifest[kind+'_input_token_counts'], strict=True):
                expected[hashlib.sha256(text.encode()).hexdigest()] = tokens
        unrelated = read_json(a.reference_inputs/'unrelated.json')
        expected[hashlib.sha256(unrelated['text'].encode()).hexdigest()] = len(unrelated['token_ids'])
        total, formal = 81, 40
    else:
        snapshot = read_json(a.reference_inputs/'prompt-snapshots/manifest.json')
        expected = {x['sha256_u32le']: x['prompt_tokens'] for x in snapshot['requests']}
        total, formal = 193, 192
    records = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_):
            pass

        def reply(self, code, payload, kind='application/json'):
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.reply(200, json.dumps({'data': [{'id': c['server']['served_model_name']}]}).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            prompt = body.get('prompt')
            warm = False
            if isinstance(prompt, list):
                value = array.array('I', prompt)
                if sys.byteorder != 'little':
                    value.byteswap()
                fingerprint = hashlib.sha256(value.tobytes()).hexdigest()
                count = expected.get(fingerprint)
            elif isinstance(prompt, str):
                fingerprint = hashlib.sha256(prompt.encode()).hexdigest()
                if a.client == 'swe':
                    warm = prompt == 'Warm up this inference service and answer briefly. This request is excluded from closed-loop metrics.'
                    count = 18 if warm else None
                else:
                    count = expected.get(fingerprint)
                    warm = body.get('stream') is False
            else:
                count, fingerprint = None, None
            matched = self.path == '/v1/completions' and count is not None
            records.append({'matched': matched, 'fingerprint': fingerprint, 'warmup': warm,
                            'expected_prompt_tokens': count, 'max_tokens': body.get('max_tokens')})
            if not matched:
                self.reply(400, b'{"error":"unexpected exact payload"}')
                return
            output = int(body['max_tokens'])
            first = {'id': 'mock', 'object': 'text_completion', 'choices': [{'index': 0, 'text': 'ok', 'finish_reason': None}], 'usage': None}
            final = {'id': 'mock', 'object': 'text_completion', 'choices': [{'index': 0, 'text': '', 'finish_reason': 'length'}],
                     'usage': {'prompt_tokens': count, 'completion_tokens': output, 'total_tokens': count+output}}
            if body.get('stream'):
                payload = ('data: '+json.dumps(first)+'\n\ndata: '+json.dumps(final)+'\n\ndata: [DONE]\n\n').encode()
                self.reply(200, payload, 'text/event-stream')
            else:
                final['choices'][0]['text'] = 'ok'
                self.reply(200, json.dumps(final).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    c['server']['port'] = server.server_address[1]
    c['paths']['output_root'] = str(evidence/'measurements')
    c['execution']['repetitions'] = 1
    save_json(evidence/'test-config.json', c)
    command = [sys.executable, '-B', str(ROOT/'scripts'/('run_'+a.client+'.py')),
               '--config', str(evidence/'test-config.json')]
    code, alive, metrics_passed = None, False, False
    try:
        with OwnedProcess(command, cwd=evidence, environment={**os.environ, 'BUNDLE_VALIDATION_ONLY': 'mock_http', 'PYTHONDONTWRITEBYTECODE': '1'},
                          log_path=evidence/'controller.log', name='test-controller', shutdown_timeout=10) as process:
            code = process.wait(900)
        # The client must not manage this independently owned HTTP server.
        http('http://127.0.0.1:'+str(server.server_address[1]), '/health', timeout=5)
        alive = True
        metrics = list((evidence/'measurements').glob('*/attempts/001/performance.json'))
        if len(metrics) == 1:
            result = read_json(metrics[0])
            metrics_passed = result['passed'] and result['requests'] == formal
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        passed = code == 0 and len(records) == total and all(x['matched'] for x in records) and alive and metrics_passed
        save_json(evidence/'mock-validation.json', {'passed': passed, 'client': a.client, 'controller_returncode': code,
                  'expected_requests_including_warmup': total, 'actual_requests': len(records),
                  'formal_requests': formal, 'actual_warmups': sum(x['warmup'] for x in records),
                  'requests': records, 'native_metrics_parsed_and_validated': metrics_passed,
                  'external_service_alive_after_client': alive, 'mock_service_closed_by_test_harness': True,
                  'synthetic_performance_must_not_be_reported_as_real': True})
    print(evidence)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
