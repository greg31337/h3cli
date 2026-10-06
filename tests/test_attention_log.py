"""Progress telemetry accepts redirected logs, redraws, and partial writes."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

from attention_run import run


class ProgressLog(unittest.TestCase):
    def test_redirected_and_terminal_progress(self):
        for delimiter in ('\n', '\r'):
            with self.subTest(delimiter=repr(delimiter)), tempfile.TemporaryDirectory() as directory:
                # Split a phase name across separate monitor reads, then leave
                # it quiet long enough to detect accidental replay of a line.
                program = (
                    "import os,time\n"
                    "assert os.environ['H3_VERBOSE']=='1'\n"
                    "os.write(1,b'deno');time.sleep(.3)\n"
                    f"os.write(1,{'ise 1/2 (0.30 s)' + delimiter!r}.encode());time.sleep(.6)\n"
                    f"os.write(1,{'audio VAE 1/1 (0.01 s)' + delimiter!r}.encode())\n"
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run('progress', [sys.executable, '-c', program], Path(directory), 10)
                self.assertEqual(result['returncode'], 0)
                observed = result['progress_observed']
                self.assertEqual(set(observed), {'denoise', 'audio VAE'})
                self.assertEqual(observed['denoise']['first_seconds'], observed['denoise']['last_seconds'])
                self.assertGreater(result['observed_denoising_seconds'], 0)


if __name__ == '__main__':
    unittest.main()
