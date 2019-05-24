import http.server as server
import cgi
import json
import os

PORT = 10000

class WebhookHandler(server.BaseHTTPRequestHandler):
  def do_POST(self):
    ctype, pdict = cgi.parse_header(self.headers['content-type'])
    print(ctype, pdict)
    
    if ctype != 'application/json':
      self.send_response(400)
      self.end_headers()
      return
    
    length = int(self.headers['content-length'])
    message = json.loads(self.rfile.read(length))
    if 'hook_id' not in message.keys():
      self.send_response(400)
      self.end_headers()
      return

    if message['hook_id'] == 112006256:
      print ('It\'s me!')
      os.system('git stash && git pull && git stash pop')
      print ('Sync complete!')
      self.send_response(200)
      self.end_headers()
      return

def run(server_class=server.HTTPServer, handler_class=WebhookHandler):
  server_address = ('', PORT)
  httpd = server_class(server_address, handler_class)
  print('Webhook server listening on port %d' % PORT)
  httpd.serve_forever()

run()
