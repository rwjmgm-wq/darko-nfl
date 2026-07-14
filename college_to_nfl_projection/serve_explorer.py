"""
Simple HTTP server to serve the College QB Explorer

Run this to avoid CORS issues when loading JSON data locally.
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path
import time
import threading

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open(f'http://localhost:{PORT}/college_qb_explorer.html')

if __name__ == '__main__':
    # Change to project directory
    project_root = Path(__file__).parent
    import os
    os.chdir(project_root)

    # Start browser opener thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Start server
    Handler = MyHTTPRequestHandler

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("=" * 70)
        print("College QB LEAF Explorer - Local Server")
        print("=" * 70)
        print(f"\nServer running at: http://localhost:{PORT}")
        print(f"Opening browser to: http://localhost:{PORT}/college_qb_explorer.html")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 70)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down server...")
            httpd.shutdown()
