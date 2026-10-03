#!/usr/bin/env python3
"""Small standard-library client for h3cli's queued video/native API."""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

class Client:
    def __init__(self, url='http://127.0.0.1:30000', token=None):
        self.url = url.rstrip('/')
        self.token = token

    def request(self, method, path, body=None):
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = 'Bearer ' + self.token
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.url + path, data=data,
                                         headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode(errors='replace')) from error

    def submit(self, body, native=False):
        return self.request('POST', '/v1/h3/jobs' if native else '/v1/videos', body)

    def wait(self, job_id, timeout=3600):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.request('GET', '/v1/h3/jobs/' + job_id)
            if job['status'] not in ('queued', 'running'):
                if job['status'] != 'completed':
                    raise RuntimeError(json.dumps(job['error']))
                return job
            time.sleep(1)
        raise TimeoutError(job_id)

    def download(self, path, output):
        if not path.startswith('/v1/') or '://' in path:
            raise ValueError('expected a relative API content/artifact URL')
        headers = {'Authorization': 'Bearer ' + self.token} if self.token else {}
        request = urllib.request.Request(self.url + path, headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response, Path(output).open('wb') as file:
            while chunk := response.read(1 << 20):
                file.write(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:30000')
    parser.add_argument('--token', default=os.environ.get('H3_SERVER_API_KEY'))
    commands = parser.add_subparsers(dest='command', required=True)
    submit = commands.add_parser('submit')
    submit.add_argument('request', type=Path, help='JSON request file')
    submit.add_argument('--native', action='store_true', help='use /v1/h3/jobs')
    submit.add_argument('--wait', action='store_true')
    submit.add_argument('--output', type=Path, help='download variant-zero video')
    for name in ('get', 'wait', 'cancel', 'delete'):
        commands.add_parser(name).add_argument('job_id')
    download = commands.add_parser('download')
    download.add_argument('path', help='relative URL from the job response')
    download.add_argument('output', type=Path)
    args = parser.parse_args()
    client = Client(args.url, args.token)
    if args.command == 'submit':
        result = client.submit(json.loads(args.request.read_text()), args.native)
        if args.wait or args.output:
            result = client.wait(result['id'])
        if args.output:
            video = next(a for a in result['h3']['artifacts']
                         if a['variant'] == 0 and a['content_type'] == 'video/mp4')
            client.download(video['url'], args.output)
    elif args.command == 'wait':
        result = client.wait(args.job_id)
    elif args.command == 'download':
        client.download(args.path, args.output)
        return
    else:
        method = {'get': 'GET', 'cancel': 'POST', 'delete': 'DELETE'}[args.command]
        path = '/v1/h3/jobs/' + args.job_id + ('/cancel' if args.command == 'cancel' else '')
        result = client.request(method, path, {} if method == 'POST' else None)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
