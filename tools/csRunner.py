#!/usr/bin/env python3
"""Minimal authenticated command/file runner for a CloudStudio box (stdlib only).

Start:  CS_TOKEN=<secret> python3 csRunner.py [port]
Auth:   every request needs header  X-Token: <secret>
Routes: GET  /health
        POST /run?timeout=600            body = bash script  -> {rc, out, err}
        POST /start?name=<job>           body = bash script  -> background job, log at jobs/<job>.log, rc at jobs/<job>.rc
        GET  /log?name=<job>&tail=4000   -> {running, rc, log}
        GET  /jobs                       -> list of jobs
        POST /put?path=<abs>&append=0|1  body = raw bytes    -> {bytes}
        GET  /get?path=<abs>             -> file bytes
        POST /kill?name=<job>            -> kill process group of job
"""
import hmac, json, os, signal, subprocess, sys, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

TOKEN = os.environ.get('CS_TOKEN', '')
if not TOKEN:
    sys.exit('CS_TOKEN env required')
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
ROOT = os.environ.get('CS_ROOT', '/workspace/casmi')
JOBS = os.path.join(ROOT, 'jobs')
os.makedirs(JOBS, exist_ok=True)
T0 = time.time()
procs = {}
lock = threading.Lock()


def jobName(q):
    n = (q.get('name') or ['job'])[0]
    if not n.replace('_', '').replace('-', '').replace('.', '').isalnum():
        raise ValueError('bad job name')
    return n


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        if not hmac.compare_digest(self.headers.get('X-Token', ''), TOKEN):
            self._send(401, {'error': 'unauthorized'})
            return False
        return True

    def _body(self):
        n = int(self.headers.get('Content-Length') or 0)
        return self.rfile.read(n) if n else b''

    def do_GET(self):
        if not self._auth():
            return
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == '/health':
                self._send(200, {'ok': True, 'uptime': round(time.time() - T0), 'load': os.getloadavg(), 'root': ROOT})
            elif u.path == '/log':
                n = jobName(q)
                tail = int((q.get('tail') or ['4000'])[0])
                logP = os.path.join(JOBS, n + '.log')
                rcP = os.path.join(JOBS, n + '.rc')
                log = ''
                if os.path.exists(logP):
                    with open(logP, 'rb') as f:
                        f.seek(0, 2)
                        sz = f.tell()
                        f.seek(max(0, sz - tail))
                        log = f.read().decode('utf-8', 'replace')
                rc = None
                if os.path.exists(rcP):
                    rc = open(rcP).read().strip()
                p = procs.get(n)
                running = p is not None and p.poll() is None
                self._send(200, {'name': n, 'running': running, 'rc': rc, 'log': log})
            elif u.path == '/jobs':
                out = []
                for f in sorted(os.listdir(JOBS)):
                    if f.endswith('.log'):
                        n = f[:-4]
                        p = procs.get(n)
                        out.append({'name': n, 'running': p is not None and p.poll() is None,
                                    'rc': open(os.path.join(JOBS, n + '.rc')).read().strip() if os.path.exists(os.path.join(JOBS, n + '.rc')) else None,
                                    'mtime': int(os.path.getmtime(os.path.join(JOBS, f)))})
                self._send(200, out)
            elif u.path == '/get':
                p = q['path'][0]
                with open(p, 'rb') as f:
                    data = f.read()
                self._send(200, data, 'application/octet-stream')
            else:
                self._send(404, {'error': 'no route'})
        except Exception as e:  # noqa
            self._send(500, {'error': repr(e)})

    def do_POST(self):
        if not self._auth():
            return
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            body = self._body()
            if u.path == '/run':
                timeout = float((q.get('timeout') or ['600'])[0])
                r = subprocess.run(['bash', '-lc', body.decode()], capture_output=True, timeout=timeout, cwd=ROOT)
                self._send(200, {'rc': r.returncode, 'out': r.stdout[-200000:].decode('utf-8', 'replace'),
                                 'err': r.stderr[-50000:].decode('utf-8', 'replace')})
            elif u.path == '/start':
                n = jobName(q)
                with lock:
                    p = procs.get(n)
                    if p is not None and p.poll() is None:
                        self._send(409, {'error': 'job running', 'name': n})
                        return
                    scriptP = os.path.join(JOBS, n + '.sh')
                    logP = os.path.join(JOBS, n + '.log')
                    rcP = os.path.join(JOBS, n + '.rc')
                    with open(scriptP, 'wb') as f:
                        f.write(body)
                    for pth in (rcP,):
                        if os.path.exists(pth):
                            os.remove(pth)
                    logF = open(logP, 'ab')
                    wrapper = f'bash {scriptP}; echo $? > {rcP}'
                    p = subprocess.Popen(['bash', '-lc', wrapper], stdout=logF, stderr=subprocess.STDOUT, cwd=ROOT,
                                         start_new_session=True)
                    procs[n] = p
                self._send(200, {'name': n, 'pid': p.pid, 'log': logP})
            elif u.path == '/kill':
                n = jobName(q)
                p = procs.get(n)
                if p is None or p.poll() is not None:
                    self._send(200, {'name': n, 'killed': False})
                    return
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                self._send(200, {'name': n, 'killed': True})
            elif u.path == '/put':
                p = q['path'][0]
                append = (q.get('append') or ['0'])[0] == '1'
                os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
                with open(p, 'ab' if append else 'wb') as f:
                    f.write(body)
                self._send(200, {'path': p, 'bytes': len(body), 'size': os.path.getsize(p)})
            else:
                self._send(404, {'error': 'no route'})
        except subprocess.TimeoutExpired:
            self._send(504, {'error': 'timeout'})
        except Exception as e:  # noqa
            self._send(500, {'error': repr(e)})


if __name__ == '__main__':
    srv = ThreadingHTTPServer(('0.0.0.0', PORT), H)
    srv.daemon_threads = True
    print(f'csRunner listening on {PORT}, root {ROOT}', flush=True)
    srv.serve_forever()
