#!/usr/bin/env python3
"""HTTP lifecycle fixture only; does not import or run an inference backend."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import array
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import threading

run = Path(os.environ['BUNDLE_RUN_DIR'])
fixture = json.loads((run/'fixture-worker.json').read_text())
fixture.update(pid=os.getpid(), run_dir=str(run), unique_id=os.environ['BUNDLE_SHM_PREFIX']+'fixture')
expected=json.loads((run/'fixture-prompts.json').read_text()) if (run/'fixture-prompts.json').exists() else None
lock=threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, value, kind='application/json'):
        data = value if isinstance(value,bytes) else json.dumps(value).encode()
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path=='/metrics':
            self.reply(b'vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n','text/plain')
            return
        self.reply({'data': [{'id': 'glm52-mixed'}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path=='/v1/completions' and expected is not None:
            prompt=body.get('prompt')
            if isinstance(prompt,list):
                values=array.array('I',prompt)
                if sys.byteorder!='little':values.byteswap()
                digest=hashlib.sha256(values.tobytes()).hexdigest()
            else:
                digest=hashlib.sha256(str(prompt).encode()).hexdigest()
            count=expected.get(digest)
            record={'matched':count is not None,'sha256':digest,'stream':body.get('stream'),'monotonic_ns':time.monotonic_ns()}
            with lock:
                with (run/'fixture-requests.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            if count is None:
                self.send_error(400,'Unexpected exact payload');return
            output=body['max_tokens']
            first={'id':'fixture','choices':[{'index':0,'text':'ok','finish_reason':None}],'usage':None}
            final={'id':'fixture','choices':[{'index':0,'text':'','finish_reason':'length'}],
                   'usage':{'prompt_tokens':count,'completion_tokens':output,'total_tokens':count+output}}
            if body.get('stream'):
                self.reply(('data: '+json.dumps(first)+'\n\ndata: '+json.dumps(final)+'\n\ndata: [DONE]\n\n').encode(),'text/event-stream')
            else:
                final['choices'][0]['text']='ok';self.reply(final)
            return
        if body.get('method')=='bundle_reap_completed':
            self.reply({'results':[{'pending_after':0,'retired_requests_after':0}]});return
        if body.get('method')=='bundle_compilation_snapshot':
            self.reply({'results':[{'pid':os.getpid(),'events':[],'installed':{'dynamo':True,'device_graph_capture':True,'triton':True},
                                   'triton_package_present':True,'scope':'CPU fixture compiler observation'}]});return
        if body['method'] != 'bundle_snapshot':
            raise ValueError('Unexpected RPC in lifecycle test')
        self.reply({'results': [fixture]})


ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
