"""Locked Mac helpers survive Homebrew shadowing and immutable snapshots."""
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE))
import host_tools
import flow


class HostToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='h3-host-tools-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()/'checkout with spaces'
        self.source = self.root/'outputs/release/source'
        self.lock(self.source, '3.0.10')
        self.lock(self.root, '3.0.11')
        self.system = self.root/'system tools'
        self.local = self.root/'outputs/setup/tools/bin'
        self.tool(self.system, '3.0.11')
        self.env = dict(os.environ, PATH=str(self.system)+os.pathsep+os.defpath)
        self.env.pop('BASH_ENV', None)

    @staticmethod
    def lock(source, version):
        path = source/'scripts/macos/lock.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(toolchain=dict(pkg_config=version))))

    @staticmethod
    def tool(directory, version):
        directory.mkdir(parents=True, exist_ok=True)
        tool = directory/'pkg-config'
        tool.write_text('#!/bin/sh\n[ "$1" = --version ] || exit 1\nprintf "%s\\n" '+shlex.quote(version)+'\n')
        tool.chmod(0o755)
        return tool

    def select(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return host_tools.macos_environment(self.source, self.root, self.env)

    def selected_version(self, env):
        return subprocess.check_output(['pkg-config', '--version'], env=env, cwd='/', text=True).strip()

    def test_original_checkout_tool_satisfies_selected_snapshot_lock(self):
        tool = self.tool(self.local, '3.0.10')
        before = dict(self.env)
        lock_before = (self.source/'scripts/macos/lock.json').read_bytes()
        files_before = set(self.root.rglob('*'))
        env = self.select()
        self.assertEqual(self.selected_version(env), '3.0.10')
        self.assertIn(str(tool.parent), env['PATH'].split(os.pathsep))
        self.assertEqual(self.env, before)
        self.assertEqual((self.source/'scripts/macos/lock.json').read_bytes(), lock_before)
        self.assertEqual(set(self.root.rglob('*')), files_before)

    def test_matching_later_path_entry_is_preferred_to_local_fallback(self):
        chosen = self.root/'configured tools'
        self.tool(chosen, '3.0.10')
        self.tool(self.local, '3.0.10')
        self.env['PATH'] += os.pathsep+str(chosen)
        env = self.select()
        self.assertEqual(self.selected_version(env), '3.0.10')
        self.assertLess(env['PATH'].index(str(chosen)), env['PATH'].index(str(self.system)))
        self.assertNotIn(str(self.local), env['PATH'])

    def test_missing_locked_version_has_actionable_error(self):
        self.tool(self.local, '3.0.9')
        with self.assertRaisesRegex(ValueError, r'requires pkg-config 3\.0\.10.*3\.0\.11.*3\.0\.9.*outputs/setup/tools/bin'):
            self.select()

    def test_failed_version_probe_is_not_accepted(self):
        tool = self.tool(self.local, '3.0.10')
        tool.write_text(tool.read_text()+'exit 1\n')
        with self.assertRaisesRegex(ValueError, 'requires pkg-config'):
            self.select()

    def test_legacy_python_reordering_keeps_locked_helper(self):
        self.tool(self.local, '3.0.10')
        python = self.root/'python install/bin'
        python.mkdir(parents=True)
        (python/'python3').symlink_to(sys.executable)
        with patch.object(host_tools.sys, 'executable', str(python/'python3')):
            env = self.select()
        # This is the unconditional PATH rewrite in older selected snapshots.
        result = subprocess.check_output(['/bin/bash', '-c',
            'export PATH="$(dirname "$(command -v python3)"):$PATH"; pkg-config --version'],
            env=env, cwd='/', text=True)
        self.assertEqual(result.strip(), '3.0.10')

    def test_bootstrap_does_not_promote_shared_python_bin_over_locked_helper(self):
        self.tool(self.local, '3.0.10')
        (self.system/'python3').symlink_to(sys.executable)
        self.env['PATH'] = str(self.local)+os.pathsep+self.env['PATH']
        with patch.object(host_tools.sys, 'executable', str(self.system/'python3')):
            env = self.select()
        result = subprocess.check_output(['/bin/bash', '-c',
            'source "$1"; h3_release_require_python; pkg-config --version',
            'h3-bootstrap', str(RELEASE/'python.sh')], env=env, cwd='/', text=True)
        self.assertEqual(result.strip(), '3.0.10')

    def test_every_local_mac_worker_receives_selected_environment(self):
        profile = dict(schema=2, version='v0.0.1', targets=['macos'], commit='a'*40, offline=False,
                       machines={'macos': dict(root=str(self.root/'host'), models='/models', model='/models/main')})
        with patch.object(flow, 'ROOT', self.root):
            runner = flow.Flow(profile, self.root/'profile.json')
        (self.root/'release').mkdir()
        (self.root/'release/worker.py').write_text('pass\n')
        marker = {'PATH': '/selected/tools'}
        for stage in ('provision', 'build-local', 'test-local', 'build-packages', 'test-packages', 'downloaded'):
            with self.subTest(stage=stage), patch.object(flow, 'ROOT', self.root), \
                 patch.object(flow, 'macos_environment', return_value=marker) as select, \
                 patch.object(flow, 'run') as run, patch.object(flow, 'read', return_value={}):
                runner.worker('macos', stage)
                select.assert_called_once_with(runner.source, self.root)
                self.assertIs(run.call_args.kwargs['env'], marker)


if __name__ == '__main__':
    unittest.main()
