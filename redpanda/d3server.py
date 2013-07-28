# -*- coding: utf-8 -*-

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import os
from __future__ import print_function

class D3HTTPRequestHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        # inserire gli static per i file
        rootdir = 'c:/xampp/htdocs/'
        try:
            if self.path.endswith('.html'):
                f = open(rootdir + self.path) #open requested file

                #send code 200 response
                self.send_response(200)

                #send header first
                self.send_header('Content-type','text-html')
                self.end_headers()

                #send file content to client
                self.wfile.write(f.read())
                f.close()
                return
            
        except IOError:
            self.send_error(404, 'file not found')
    

# il run va spostato dentro RedPanda così da avviare 
def run():
    print('d3 http server is starting...')
    server_address = ('127.0.0.1', 80)
    httpd = HTTPServer(server_address, D3HTTPRequestHandler)
    print('d3 http server is now running...')
    httpd.serve_forever()
