"""Opt-in local CLI smoke: self-authored MCP tools inherit Bash containment.

Run on Linux with Python 3 after `cargo build`. Uses a local scripted provider,
no external API. Scratch files are kept outside /tmp because the Bash policy
intentionally grants temporary-directory writes. All scratch data is removed.
"""
import http.server
import json
import os
import pathlib
import subprocess
import tempfile
import threading

repository = pathlib.Path(__file__).resolve().parents[2]
binary = pathlib.Path(os.environ.get('CLAUX_SMOKE_BINARY', repository / 'target/debug/claux'))
example = (repository / 'examples/live-mcp/server.py').read_text()
with tempfile.TemporaryDirectory(prefix='.claux-live-mcp-', dir=repository) as scratch:
    root = pathlib.Path(scratch)
    workspace = root / 'workspace'
    workspace.mkdir()
    home = root / 'home'
    home.mkdir()
    inside = workspace / "inside-'$().txt"
    outside = root / 'outside.txt'
    server_path = workspace / "server-'$().py"
    # The same sandbox must hold during initialization and later tool calls.
    probe = f'''\nfrom pathlib import Path
Path({str(inside)!r}).write_text('inside allowed')
try:
    Path({str(outside)!r}).write_text('escaped')
except PermissionError:
    pass
else:
    raise RuntimeError('outside write unexpectedly permitted')
'''
    source = example.replace('for line in sys.stdin:', probe + '\nfor line in sys.stdin:')
    source = source.replace("        text = request['params']['arguments']['text']", ''.join('        '+line+'\n' for line in probe.strip().splitlines()) + "        text = request['params']['arguments']['text']")
    steps = [
        [('Write', {'file_path':str(server_path),'content':source}), ('Write', {'file_path':str(workspace/'.mcp.json'),'content':json.dumps({'mcpServers':{'self':{'command':'python3','args':[str(server_path)]}}})})],
        [('McpManage',{'action':'load','name':'self'})],
        [('mcp__self__slugify',{'text':'Sandbox Verified'})],
        [('McpManage',{'action':'unload','name':'self'})],
        [],
    ]
    expected_schemas = [False, False, True, True, False]
    received = []
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            i = len(received)
            received.append(request)
            names = [t['function']['name'] for t in request['tools']]
            assert ('mcp__self__slugify' in names) == expected_schemas[i], (i,names)
            delta = {'role':'assistant'}
            if steps[i]:
                delta['tool_calls'] = [{'index':j,'id':f'call-{i}-{j}','type':'function','function':{'name':name,'arguments':json.dumps(args)}} for j,(name,args) in enumerate(steps[i])]
            else: delta['content'] = 'verified'
            payload = 'data: '+json.dumps({'id':'fixture','choices':[{'index':0,'delta':delta,'finish_reason':None}]})+'\n\n'
            payload += 'data: '+json.dumps({'id':'fixture','choices':[{'index':0,'delta':{},'finish_reason':'tool_calls' if steps[i] else 'stop'}]})+'\n\ndata: [DONE]\n\n'
            self.send_response(200)
            self.send_header('Content-Type','text/event-stream')
            self.send_header('Content-Length',str(len(payload.encode())))
            self.end_headers()
            self.wfile.write(payload.encode())
    server = http.server.HTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    config = root/'config.toml'
    config.write_text(f'''default_model = "test"
[providers.fixture]
type = "openai"
base_url = "http://127.0.0.1:{server.server_port}/v1"
api_key_env = "CLAUX_FIXTURE_KEY"
[model_profiles.test]
provider = "fixture"
model = "fixture"
context_window = 32000
''')
    env = dict(os.environ, HOME=str(home), XDG_CONFIG_HOME=str(home/'config'), XDG_DATA_HOME=str(home/'data'), CLAUX_FIXTURE_KEY='local-only')
    try:
        result = subprocess.run([str(binary),'--config',str(config),'--trust-project','--bash-fs-policy','workspace_write','--permission-mode','bypass','--transcript',str(workspace/'trace.json'),'-p','Create and use a tool'],cwd=workspace,env=env,capture_output=True,text=True,timeout=45)
        assert result.returncode == 0, result.stderr
        assert len(received) == 5, (len(received),result.stderr)
        assert inside.read_text() == 'inside allowed'
        assert not outside.exists(), 'server escaped containment'
        trace = json.loads((workspace/'trace.json').read_text())
        encoded = json.dumps(trace)
        assert 'sandbox-verified' in encoded, encoded
        assert 'McpManage failed' not in encoded, encoded
        # Start a new Claux process with the generated .mcp.json already on
        # disk. Project servers must remain contained at startup, not just
        # during the original live load. Exercise initialization and a call.
        received.clear()
        steps[:] = [[('mcp__self__slugify', {'text':'Restart Verified'})], []]
        expected_schemas[:] = [True, True]
        restart_trace = workspace/'restart-trace.json'
        restarted = subprocess.run([str(binary),'--config',str(config),'--trust-project','--bash-fs-policy','workspace_write','--permission-mode','bypass','--transcript',str(restart_trace),'-p','Use the tool'],cwd=workspace,env=env,capture_output=True,text=True,timeout=45)
        assert restarted.returncode == 0, restarted.stderr
        assert len(received) == 2, (len(received),restarted.stderr)
        assert not outside.exists(), 'server escaped containment after restart'
        restarted_text = json.dumps(json.loads(restart_trace.read_text()))
        assert 'restart-verified' in restarted_text, restarted_text
        print('PASS: live-loaded and restarted project MCP servers remain contained during initialization and invocation; schema refresh and literal argument handling work.')
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
