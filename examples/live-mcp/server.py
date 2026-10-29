"""Tiny stdio MCP server for the live-load spike; Python standard library only."""
import json
import re
import sys

for line in sys.stdin:
    request = json.loads(line)
    if 'id' not in request:
        continue
    method = request.get('method')
    if method == 'initialize':
        result = {
            'protocolVersion': request['params']['protocolVersion'],
            'capabilities': {'tools': {}},
            'serverInfo': {'name': 'local-slug', 'version': '1'},
        }
    elif method == 'tools/list':
        result = {'tools': [{
            'name': 'slugify', 'description': 'Convert text to an ASCII slug.',
            'inputSchema': {'type': 'object', 'properties': {'text': {'type': 'string'}},
                            'required': ['text'], 'additionalProperties': False},
            'annotations': {'readOnlyHint': True},
        }]}
    elif method == 'tools/call' and request['params']['name'] == 'slugify':
        text = request['params']['arguments']['text']
        slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
        result = {'content': [{'type': 'text', 'text': slug}]}
    elif method == 'ping':
        result = {}
    else:
        print(json.dumps({'jsonrpc': '2.0', 'id': request['id'],
                          'error': {'code': -32601, 'message': 'Unknown method'}}), flush=True)
        continue
    print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}), flush=True)
