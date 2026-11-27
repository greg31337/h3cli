"""Exercise interpreter selection with shadowed Python and a simulated Homebrew."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

RELEASE = Path(__file__).resolve().parents[1]


class PythonBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='h3-release-python-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.checkout = self.root/'checkout with spaces'
        release = self.checkout/'release'
        release.mkdir(parents=True)
        for name in ('release.sh', 'prerequisites.sh', 'python.sh'):
            shutil.copy2(RELEASE/name, release/name)
        self.tools = self.root/'tools'
        self.old = self.root/'old python'
        self.good = self.root/'supported python'
        self.brew = self.root/'homebrew prefix'
        for path in (self.tools, self.old, self.good, self.brew/'bin'):
            path.mkdir(parents=True)
        for name in ('bash', 'dirname'):
            (self.tools/name).symlink_to(shutil.which(name))
        self.script(self.tools/'uname', 'if [[ "$1" == -s ]]; then echo Darwin; else echo arm64; fi')
        for name in ('git', 'pkg-config', 'xcrun', 'gh', 'minisign'):
            self.script(self.tools/name, 'exit 0')
        self.script(self.old/'python3', 'echo "old Python must not run release code" >&2; exit 91')
        self.script(self.good/'python3', 'export H3_TEST_PYTHON=supported\nexec '+shlex.quote(sys.executable)+' "$@"')
        self.script(self.tools/'brew', '''case "$1" in
  --prefix) printf '%s\\n' "$H3_TEST_BREW_PREFIX" ;;
  install)
    printf '%s\\n' "$@" > "$H3_TEST_INSTALL_LOG"
    if [[ "${H3_TEST_INSTALL_PYTHON-}" == 1 ]]; then
      /bin/cp "$H3_TEST_GOOD_PYTHON" "$H3_TEST_BREW_PREFIX/bin/python3"
    fi ;;
  *) exit 92 ;;
esac''')
        # Both the entry point and doctor launch a child by name after chdir.
        (release/'flow.py').write_text('''import json, os, subprocess, sys
print(json.dumps(dict(parent=os.environ['H3_TEST_PYTHON'], args=sys.argv[1:],
    child=subprocess.check_output(['python3', '-c',
        'import os; print(os.environ["H3_TEST_PYTHON"])'], cwd='/', text=True).strip())))
''')
        scripts = self.checkout/'scripts'
        scripts.mkdir()
        self.script(scripts/'build_macos.sh', 'exec python3 "$(dirname "$0")/../release/flow.py" "$@"')
        self.env = dict(os.environ, PATH=os.pathsep.join(map(str, (self.tools, self.old, self.good))),
                        H3_TEST_BREW_PREFIX=str(self.brew), H3_TEST_INSTALL_LOG=str(self.root/'install.log'),
                        H3_TEST_GOOD_PYTHON=str(self.good/'python3'))
        self.env.pop('BASH_ENV', None)

    @staticmethod
    def script(path, body):
        path.write_text('#!/bin/bash\nset -eu\n'+body+'\n')
        path.chmod(0o755)

    def run_script(self, name, *args):
        return subprocess.run(['/bin/bash', str(self.checkout/'release'/name), *args],
                              cwd=self.root, env=self.env, capture_output=True, text=True)

    def assert_selected(self, result, expected='supported'):
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        row = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual((row['parent'], row['child']), (expected, expected))
        return row

    def test_entrypoint_skips_old_python_and_propagates_to_children(self):
        args = ('configure', '--version', 'v0.9.2', '--notes', 'a path with spaces')
        row = self.assert_selected(self.run_script('release.sh', *args))
        self.assertEqual(row['args'], list(args))
        self.assertFalse((self.root/'install.log').exists())

    def test_entrypoint_preserves_compatible_environment_first_on_path(self):
        venv = self.root/'virtual environment'
        venv.mkdir()
        self.script(venv/'python3', 'export H3_TEST_PYTHON=venv\nexec '+shlex.quote(sys.executable)+' "$@"')
        self.env['PATH'] = str(venv)+os.pathsep+self.env['PATH']
        self.assert_selected(self.run_script('release.sh', '--help'), 'venv')

    def test_entrypoint_finds_homebrew_python_outside_path(self):
        shutil.copy2(self.good/'python3', self.brew/'bin/python3')
        self.env['PATH'] = os.pathsep.join(map(str, (self.tools, self.old)))
        self.assert_selected(self.run_script('release.sh', '--help'))

    def test_missing_compatible_python_explains_bootstrap_without_installing(self):
        self.env['PATH'] = os.pathsep.join(map(str, (self.tools, self.old)))
        for name, args in (('release.sh', ('--help',)),
                           ('prerequisites.sh', ('--platform', 'macos'))):
            with self.subTest(name=name):
                result = self.run_script(name, *args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('Python 3.10+', result.stderr)
                self.assertIn('step1-prerequisites.sh --install', result.stderr)
                self.assertFalse((self.root/'install.log').exists())

    def test_prerequisites_doctor_uses_selected_python(self):
        self.assert_selected(self.run_script('prerequisites.sh', '--platform', 'macos'))

    def test_install_does_not_reinstall_shadowed_supported_python(self):
        self.assert_selected(self.run_script('prerequisites.sh', '--platform', 'macos', '--install'))
        self.assertFalse((self.root/'install.log').exists())

    def test_install_discovers_new_python_outside_path(self):
        self.env['PATH'] = os.pathsep.join(map(str, (self.tools, self.old)))
        self.env['H3_TEST_INSTALL_PYTHON'] = '1'
        self.assert_selected(self.run_script('prerequisites.sh', '--platform', 'macos', '--install'))
        self.assertEqual((self.root/'install.log').read_text().splitlines(), ['install', 'python'])

    def test_dry_run_needs_no_python_or_homebrew_and_installs_nothing(self):
        self.env['PATH'] = str(self.tools)
        (self.tools/'brew').unlink()
        result = self.run_script('prerequisites.sh', '--platform', 'macos', '--install', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('brew install', result.stdout)
        self.assertFalse((self.root/'install.log').exists())


if __name__ == '__main__':
    unittest.main()
