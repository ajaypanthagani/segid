"""Local server for the paint screen.  The page sends what you did (strokes, cuts, drags) once per frame; the server
applies it to the paint world, shows the model the resulting frame, and returns the image plus what the model thinks.

    PYTHONPATH=. .venv/bin/python web/server.py                      # then open http://localhost:8765
    PYTHONPATH=. .venv/bin/python web/server.py --memory mem.npz     # keep what the model learns between runs
"""
import argparse
import json
import os
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from objectlearn.paint import PaintApp

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))


class App:
    def __init__(self, memory=None, patch=10, mode="image", latent=False):
        self.lock = threading.Lock()
        self.sent_act = -1
        # the patch representation needs far smaller matrices and a slower refresh when the memory is large
        self.paint = PaintApp(memory=memory, s=patch, cap=(1500 if mode == "patch" else 2500) if patch < 10 else 2000,
                              mode=mode, refresh_every=10 if mode == "patch" else 3, latent=latent)

    def tick(self, body):
        data = json.loads(body or b"{}")
        actions = data.get("actions", [])
        if not isinstance(actions, list):
            raise ValueError("actions must be a list")
        with self.lock:
            t0 = time.time()
            out = self.paint.tick(actions, t0)
            out["stats"] = dict(out["stats"], ms=int(1000 * (time.time() - t0)))
            ses = self.paint.session
            aid = getattr(ses, "_act_id", 0)
            if ses.latent_on and aid != self.sent_act:        # which stored entries the latest reading leaned on, sent when it changes
                self.sent_act = aid
                out["latent_act"] = ses.latent_active()
            return out


def explain(app, body):
    d = json.loads(body or b"{}")
    with app.lock:
        return app.paint.explain(d["x"], d["y"]) or dict(found=False, why="nothing yet")


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(os.path.join(ROOT, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif self.path == "/latent":
                with app.lock:
                    c = app.paint.session.latent_cloud()
                    if c is not None:
                        c["act"] = app.paint.session.latent_active()
                self._send(200, c if c is not None else {"n": -1})
            elif self.path == "/health":
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n) if n else b""
            try:
                if self.path == "/tick":
                    self._send(200, app.tick(body))
                elif self.path == "/entry":
                    with app.lock:
                        e = app.paint.session.latent_entry(json.loads(body or b"{}").get("idx", -1))
                    self._send(200 if e else 404, e or {"error": "no such entry"})
                elif self.path == "/explain":
                    self._send(200, explain(app, body))
                else:
                    self._send(404, {"error": "not found"})
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})

        def log_message(self, *a):
            pass
    return Handler


def serve(port=8765, memory=None, host="127.0.0.1", patch=10, mode="image", latent=False):
    app = App(memory, patch, mode, latent)
    return ThreadingHTTPServer((host, port), make_handler(app)), app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=10, help="patch spacing in pixels; 5 suits small parts such as a bottle cap (slower)")
    ap.add_argument("--mode", choices=["image", "patch"], default="image", help="descriptor: from the picture, or from patches alone (cheaper)")
    ap.add_argument("--latent", action="store_true", help="use the latent space of physical traits (needs --mode patch); --memory is then its file")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--memory", default=None, help="file to load from and save to after each learning event")
    a = ap.parse_args()
    if a.latent and a.mode != "patch":
        ap.error("--latent needs --mode patch")
    srv, app = serve(a.port, a.memory, patch=a.patch, mode=a.mode, latent=a.latent)
    print(f"open http://localhost:{a.port}   (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.paint.session.save()
