#!/usr/bin/env python3
"""电阻检测 - 图片保存服务
保存裁剪后的图片，支持下载最新/历史图片
"""

import os
import time
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detection_results')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# 图片元数据
META_FILE = os.path.join(UPLOAD_DIR, 'meta.json')

def load_meta():
    if os.path.exists(META_FILE):
        with open(META_FILE, 'r') as f:
            return json.load(f)
    return []

def save_meta(meta):
    with open(META_FILE, 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/save' or parsed.path == '/':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            if not body:
                self.send_json(400, {'error': '空文件'})
                return

            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            uid = str(int(time.time() * 1e6))[-6:]
            filename = f'{ts}_{uid}.png'
            filepath = os.path.join(UPLOAD_DIR, filename)

            with open(filepath, 'wb') as f:
                f.write(body)

            # 获取图片尺寸
            w, h = 0, 0
            try:
                from PIL import Image
                with Image.open(filepath) as img:
                    w, h = img.size
            except:
                pass

            meta = load_meta()
            record = {
                'filename': filename,
                'width': w,
                'height': h,
                'size': len(body),
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            meta.append(record)
            save_meta(meta)

            self.send_json(200, {
                'ok': True,
                'filename': filename,
                'width': w,
                'height': h,
                'size': len(body)
            })

        elif parsed.path == '/save_result':
            # 保存检测结果：裁剪子图到 good/bad 文件夹
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                import base64, io
                from PIL import Image
                data = json.loads(body)
                image_data_url = data.get('image', '')
                results = data.get('results', [])

                if not image_data_url.startswith('data:'):
                    self.send_json(400, {'error': '无效的图片数据'})
                    return

                # 解码图片
                header, b64 = image_data_url.split(',', 1)
                img_bytes = base64.b64decode(b64)
                img = Image.open(io.BytesIO(img_bytes))
                W, H = img.size

                # 子区域坐标
                regions = [
                    ('smd_components', [0.01, 0.01, 0.51, 0.14]),
                    ('main_chip', [0.21, 0.54, 0.74, 0.77]),
                    ('bottom_chip', [0.18, 0.81, 0.60, 0.95])
                ]

                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                saved = []
                for i, (label, box) in enumerate(regions):
                    verdict = results[i]['verdict'] if i < len(results) else 'good'
                    # 每个区域一个文件夹
                    folder = os.path.join(SAVE_DIR, label)
                    os.makedirs(folder, exist_ok=True)

                    x1, y1, x2, y2 = [int(v) for v in [box[0]*W, box[1]*H, box[2]*W, box[3]*H]]
                    sub = img.crop((x1, y1, x2, y2))
                    fname = f'{ts}_{verdict}.png'
                    sub.save(os.path.join(folder, fname))
                    saved.append({'label': label, 'verdict': verdict, 'file': fname, 'folder': folder})

                self.send_json(200, {'ok': True, 'saved': saved})
            except Exception as e:
                self.send_json(500, {'error': str(e)})
        else:
            self.send_json(404, {'error': 'not found'})

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/latest':
            meta = load_meta()
            if meta:
                latest = meta[-1]
                self.send_json(200, latest)
            else:
                self.send_json(404, {'error': '没有保存的图片'})

        elif path == '/list':
            meta = load_meta()
            self.send_json(200, meta)

        elif path.startswith('/download/'):
            filename = os.path.basename(path)
            filepath = os.path.join(UPLOAD_DIR, filename)
            if os.path.exists(filepath):
                self.send_file(filepath, filename)
            else:
                self.send_json(404, {'error': '文件不存在'})

        elif path == '/' or path == '/latest/download':
            meta = load_meta()
            if meta:
                latest = meta[-1]
                filepath = os.path.join(UPLOAD_DIR, latest['filename'])
                if os.path.exists(filepath):
                    self.send_file(filepath, latest['filename'])
                    return
            self.send_json(404, {'error': '没有保存的图片'})

        else:
            self.send_json(404, {'error': 'not found'})

    def send_json(self, code, data):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def send_file(self, filepath, filename):
        import mimetypes
        ext = os.path.splitext(filename)[1]
        mime = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        size = os.path.getsize(filepath)

        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(size))
        self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        with open(filepath, 'rb') as f:
            self.wfile.write(f.read())

    def log_message(self, format, *args):
        pass  # 静默日志


if __name__ == '__main__':
    port = 8001
    server = HTTPServer(('0.0.0.0', port), Handler)
    print(f'Save server running on port {port}')
    print(f'Upload dir: {UPLOAD_DIR}')
    server.serve_forever()
