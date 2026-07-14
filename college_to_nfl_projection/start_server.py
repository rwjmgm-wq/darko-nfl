#!/usr/bin/env python
"""
Start a local HTTP server to view the QB Projection Visualizer
"""
import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

PORT = 8001

# Change to the script's directory
os.chdir(Path(__file__).parent)

Handler = http.server.SimpleHTTPRequestHandler

print("="*60)
print("QB PROJECTION VISUALIZER - Local Server")
print("="*60)
print(f"\nStarting server at http://localhost:{PORT}")
print(f"Opening browser to http://localhost:{PORT}/college_qb_explorer.html")
print("\nPress Ctrl+C to stop the server")
print("="*60)

# Open browser
webbrowser.open(f'http://localhost:{PORT}/college_qb_explorer.html')

# Start server
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"\nServer is running on port {PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        httpd.shutdown()
