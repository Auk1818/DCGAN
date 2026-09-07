"""DCGAN训练实时监控服务器

用法:
    python monitor.py

启动后浏览器访问 http://localhost:8765 即可查看训练进度。
页面每 8 秒自动刷新，显示最新 epoch 的生成图片和 Loss 曲线。
按 Ctrl+C 停止服务器。
"""

import os
import re
import sys
import argparse
import http.server
import socketserver

DEFAULT_RESULT_DIR = r'C:\dcgan_result'
DEFAULT_PORT = 8765
DEFAULT_EPOCHS = 40
REFRESH_INTERVAL = 8  # 页面自动刷新间隔（秒）


def get_latest_epoch(result_dir):
    """扫描结果目录，获取最新 epoch 编号"""
    pattern = re.compile(r'generated_epoch(\d+)\.png')
    max_epoch = 0
    if os.path.exists(result_dir):
        for f in os.listdir(result_dir):
            m = pattern.match(f)
            if m:
                max_epoch = max(max_epoch, int(m.group(1)))
    return max_epoch


def build_html(result_dir, total_epochs):
    """生成监控页面 HTML"""
    current_epoch = get_latest_epoch(result_dir)
    progress_pct = (current_epoch / total_epochs * 100) if total_epochs > 0 else 0

    generated_img = f'generated_epoch{current_epoch}.png' if current_epoch > 0 else 'sample_data.png'
    loss_img = f'loss_curve_epoch{current_epoch}.png' if current_epoch > 0 else ''
    is_done = current_epoch >= total_epochs
    status_text = '训练完成' if is_done else '训练进行中'
    dot_color = '#00d4ff' if is_done else '#00ff88'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>DCGAN训练监控</title>
<meta http-equiv="refresh" content="{REFRESH_INTERVAL}">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', sans-serif;
    background: #0f0f1a;
    color: #e0e0e0;
    padding: 20px;
}}
h1 {{
    text-align: center;
    color: #00d4ff;
    margin-bottom: 20px;
    font-size: 24px;
}}
.dashboard {{
    max-width: 1200px;
    margin: 0 auto;
}}
.progress-section {{
    background: #1a1a2e;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid #2a2a4a;
}}
.progress-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}}
.progress-bar {{
    width: 100%;
    height: 24px;
    background: #2a2a4a;
    border-radius: 12px;
    overflow: hidden;
}}
.progress-fill {{
    height: 100%;
    background: linear-gradient(90deg, #00d4ff, #00ff88);
    border-radius: 12px;
    transition: width 0.5s ease;
}}
.epoch-info {{
    font-size: 18px;
    color: #00d4ff;
}}
.pct {{
    font-size: 14px;
    color: #888;
}}
.images-row {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}}
.image-card {{
    flex: 1;
    min-width: 300px;
    background: #1a1a2e;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #2a2a4a;
}}
.image-card h2 {{
    font-size: 16px;
    color: #aaa;
    margin-bottom: 12px;
}}
.image-card img {{
    width: 100%;
    border-radius: 8px;
    background: #111;
}}
.status {{
    text-align: center;
    margin-top: 16px;
    color: #666;
    font-size: 13px;
}}
.dot {{
    display: inline-block;
    width: 8px; height: 8px;
    background: {dot_color};
    border-radius: 50%;
    margin-right: 6px;
    animation: blink 1.5s infinite;
}}
@keyframes blink {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
}}
</style>
</head>
<body>
<div class="dashboard">
    <h1>DCGAN 漫画头像生成 - 训练监控</h1>

    <div class="progress-section">
        <div class="progress-header">
            <span class="epoch-info">Epoch: {current_epoch} / {total_epochs} ({status_text})</span>
            <span class="pct">{progress_pct:.1f}%</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress_pct}%"></div>
        </div>
    </div>

    <div class="images-row">
        <div class="image-card">
            <h2>生成图片 (Epoch {current_epoch})</h2>
            <img src="{generated_img}?t={current_epoch}" alt="generated">
        </div>
        <div class="image-card">
            <h2>Loss 曲线 (Epoch {current_epoch})</h2>
            <img src="{loss_img}?t={current_epoch}" alt="loss curve">
        </div>
    </div>

    <div class="status">
        <span class="dot"></span>
        页面每 {REFRESH_INTERVAL} 秒自动刷新 | {status_text} | 最后更新: <span id="time"></span>
    </div>
</div>
<script>
document.getElementById('time').textContent = new Date().toLocaleTimeString();
</script>
</body>
</html>"""


class MonitorHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, result_dir=DEFAULT_RESULT_DIR, total_epochs=DEFAULT_EPOCHS, **kwargs):
        self.result_dir = result_dir
        self.total_epochs = total_epochs
        super().__init__(*args, directory=result_dir, **kwargs)

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            content = build_html(self.result_dir, self.total_epochs).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description='DCGAN训练实时监控服务器')
    parser.add_argument('--dir', default=DEFAULT_RESULT_DIR, help=f'结果目录路径 (默认: {DEFAULT_RESULT_DIR})')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'服务器端口 (默认: {DEFAULT_PORT})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help=f'总训练轮数 (默认: {DEFAULT_EPOCHS})')
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"错误: 结果目录不存在: {args.dir}")
        print("请确认训练脚本已启动并生成了输出文件。")
        sys.exit(1)

    handler = lambda *a, **kw: MonitorHandler(*a, result_dir=args.dir, total_epochs=args.epochs, **kw)

    with socketserver.TCPServer(('', args.port), handler) as httpd:
        print(f"监控服务器已启动: http://localhost:{args.port}")
        print(f"监控目录: {args.dir}")
        print(f"总训练轮数: {args.epochs}")
        print("按 Ctrl+C 停止")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止")


if __name__ == '__main__':
    main()
