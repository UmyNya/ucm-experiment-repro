#!/usr/bin/env python3
"""CPU integration check: real AISBench against a strict local fixture, never an LLM."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import new_run, save_json, read_json
from configuration import load, compile_plan
from workloads import ais_benchmark_command
from processes import OwnedProcess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/experiment.json')
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    c = load(args.config, ['paths.aisbench_python=' + args.python])
    manifest = read_json(args.inputs / 'manifest.json')
    texts = [json.loads(line)['question'] for line in (args.inputs/'formal.jsonl').read_text().splitlines()]
    expected = dict(zip(texts, manifest['formal_input_token_counts'], strict=True))
    records = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_):
            pass

        def do_GET(self):
            payload = json.dumps({'object': 'list', 'data': [{'id': c['server']['served_model_name'], 'object': 'model'}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            value = json.loads(body)
            prompt = value.get('prompt')
            if prompt not in expected:
                data = b'{"error":"mock received an unexpected prompt"}'
                self.send_response(400)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                records.append({'matched': False, 'path': self.path})
                return
            records.append({'matched': True, 'path': self.path,
                            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
            count = int(value['max_tokens'])
            first = {'id': 'mock', 'object': 'text_completion', 'model': c['server']['served_model_name'],
                     'choices': [{'index': 0, 'text': 'ok', 'finish_reason': None}],
                     'usage': None}
            final = {'id': 'mock', 'object': 'text_completion', 'model': c['server']['served_model_name'],
                     'choices': [{'index': 0, 'text': '', 'finish_reason': 'length'}],
                     'usage': {'prompt_tokens': expected[prompt], 'completion_tokens': count,
                               'total_tokens': expected[prompt] + count}}
            if value.get('stream'):
                payload = ('data: ' + json.dumps(first) + '\n\ndata: ' + json.dumps(final) + '\n\ndata: [DONE]\n\n').encode()
                content_type = 'text/event-stream'
            else:
                final['choices'][0]['text'] = 'ok'
                payload, content_type = json.dumps(final).encode(), 'application/json'
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    c['server']['port'] = server.server_address[1]
    run = new_run(c, 'aisbench_n40_performance', 'mock-client-validation')
    plan = compile_plan(c, 'aisbench_n40_performance', run)
    plan['validation_only'] = 'CPU mock HTTP server; all latency/throughput values are synthetic'
    save_json(run/'plan.json', plan)
    command = ais_benchmark_command(plan, run, args.inputs.resolve())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
                   'TRANSFORMERS_OFFLINE': '1', 'HF_HOME': str(run/'scratch/hf')}
    code = None
    try:
        with OwnedProcess(command, cwd=run, environment=environment, log_path=run/'mock-client.log',
                          name='aisbench', shutdown_timeout=5) as process:
            code = process.wait(300)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        save_json(run/'mock-validation.json', {'scope': 'mock only; no model inference', 'returncode': code,
                   'requests': records, 'expected_count': len(texts),
                   'passed': code == 0 and len(records) == len(texts) and all(x['matched'] for x in records),
                   'synthetic_performance_must_not_be_reported_as_real': True})
    print(run)
    if code or len(records) != len(texts) or not all(x['matched'] for x in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
