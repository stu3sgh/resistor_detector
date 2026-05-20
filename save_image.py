#!/usr/bin/env python3
"""Minimal image save endpoint for resistor detector."""
import os, sys, json, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

UPLOAD_DIR = '/tmp/resistor_uploads'

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != '/save':
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        fname = f'{ts}.png'
        fpath = os.path.join(UPLOAD_DIR, fname)
        with open(fpath, 'wb') as f:
            f.write(body)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'ok': True, 'file': fname}).encode())

    def do_GET(self):
        if self.path == '/list':
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            files = sorted(os.listdir(UPLOAD_DIR), reverse=True)[:20]
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'files': files}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass  # suppress logs

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    HTTPServer(('127.0.0.1', port), Handler).serve_forever()
