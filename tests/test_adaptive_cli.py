"""Policy errors must precede model loading, regardless of flag ordering."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import subprocess
import unittest


class Policy(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(["bin/h3cli", "-d", "/missing-adaptive-model", "-p", "test", "--steps", "6", *args],
                              capture_output=True, text=True)

    def test_invalid_values(self):
        for args, message in [(("--adaptive-cache", "yes"), "adaptive-cache must"),
                              (("--subblock-sparsity", "nan"), "invalid SubBlock sparsity"),
                              (("--subblock-sparsity", "1"), "finite and in"),
                              (("--subblock-sparsity", ".75"), "requires --cuda-attention")]:
            with self.subTest(args=args):
                r = self.run_cli(*args)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(message, r.stderr)

    def test_budget(self):
        for value in ("0", "-1", "+1", "auto", "4096M", "4.5", " 4", "17592186044416"):
            r=self.run_cli("--adaptive-cache-max-mib",value)
            self.assertNotEqual(r.returncode,0);self.assertIn("positive decimal integer",r.stderr)
        for args in [("--adaptive-cache-max-mib","4096","--adaptive-cache","off"),
                     ("--adaptive-cache","off","--adaptive-cache-max-mib","4096")]:
            r=self.run_cli(*args);self.assertIn("requires an enabled",r.stderr)
        r=self.run_cli("--adaptive-cache","conservative","--adaptive-cache-max-mib","1")
        self.assertIn("minimum --adaptive-cache-max-mib",r.stderr)

    def test_budget_resume_and_decode(self):
        r=subprocess.run(["bin/h3cli","--resume-sampler-state","/missing-budget-state", "--adaptive-cache-max-mib","4096"],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertNotIn("generation-changing",r.stderr)
        self.assertNotIn("requires an enabled",r.stderr);self.assertIn("cannot open state",r.stderr)
        for flag in ("--decode-av-state","--decode-still-latent"):
            r=subprocess.run(["bin/h3cli",flag,"/missing-state","--adaptive-cache-max-mib","4096"],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn("decode-only",r.stderr)

    def test_reference_policy(self):
        for feature in [("--adaptive-cache","conservative","--cuda-denoise-quant","fp8"),
                        ("--cuda-attention","subblock","--cuda-denoise-quant","fp8")]:
            for args in [(*feature,"--ref-image","inputs/1.jpg"),("--ref-image","inputs/1.jpg",*feature)]:
                r=self.run_cli(*args);self.assertIn("not qualified",r.stderr)
        r=self.run_cli("--cuda-attention","subblock","--ref-audio","inputs/1.jpg")
        self.assertIn("requires an image or video",r.stderr)
        r=self.run_cli("--cuda-attention","subblock","--ref-image","inputs/1.jpg")
        self.assertNotIn("not qualified",r.stderr)
        for flags in [("--ref-video","tests/fixtures/cuda-reference/reference.mp4"),
                      ("--ref-image","inputs/1.jpg","--ref-audio","inputs/1.jpg")]:
            r=self.run_cli("--cuda-attention","subblock",*flags)
            self.assertNotIn("not qualified",r.stderr);self.assertNotIn("image-only",r.stderr)

    def test_conflicts(self):
        for flag, value in [("--reuse", "2"), ("--core-reuse", "4"), ("--layers", "45"),
                            ("--cuda-attention", "sol")]:
            for args in [("--adaptive-cache", "aggressive", flag, value),
                         (flag, value, "--adaptive-cache", "aggressive")]:
                with self.subTest(args=args):
                    r = self.run_cli(*args)
                    self.assertNotEqual(r.returncode, 0)
                    self.assertIn("require CUDA video", r.stderr)

    def test_continuation_option_order(self):
        features = [[], ["--adaptive-cache", "conservative", "--adaptive-cache-warmup", "2"],
                    ["--adaptive-cache", "aggressive", "--adaptive-cache-warmup", "2"],
                    ["--cuda-attention", "subblock", "--subblock-warmup", "3"]]
        features += [features[1] + features[3], features[2] + features[3]]
        for mode in ("hard", "bridge"):
            continuation = ["--continue-from", "/missing-continuation-source", "--continue-mode", mode]
            for feature in features:
                for args in (continuation + feature, feature + continuation):
                    with self.subTest(args=args):
                        result = self.run_cli(*args)
                        self.assertNotEqual(result.returncode, 0)
                        # CUDA accepts the policy and attempts to load the source;
                        # Metal rejects approximation at its normal backend guard.
                        self.assertTrue("cannot open h3av state" in result.stderr or
                                        (feature and "Metal is unsupported" in result.stderr), result.stderr)

    def test_quantized_cache_policy(self):
        for mode in ('fp8','nvfp4'):
            for flags in [("--adaptive-cache","aggressive"),
                          ("--cuda-attention","subblock","--adaptive-cache","conservative")]:
                for args in [(*flags,"--cuda-denoise-quant",mode),("--cuda-denoise-quant",mode,*flags)]:
                    with self.subTest(args=args):
                        r=self.run_cli(*args)
                        self.assertNotEqual(r.returncode,0)
                        self.assertIn('requires conservative cache and dense attention',r.stderr)

    def test_custom_controls(self):
        bad={"--adaptive-cache-threshold":["", "nan", "inf", "-0", "+.04", "0x1p-4", "1.00000000000000000001", "1e-99", " 0.1", "0.1x"],
             "--adaptive-cache-max-hits":["", "0", "17", "-1", "+1", "1.5", "1e0", "9999999999999999999999"]}
        for flag,values in bad.items():
            for value in values:
                with self.subTest(flag=flag,value=value):
                    r=self.run_cli("--adaptive-cache","conservative",flag,value)
                    self.assertEqual(r.returncode,2);self.assertIn(flag[2:],r.stderr)
        for flag,value in [("--adaptive-cache-threshold","0"),("--adaptive-cache-threshold","4e-2"),("--adaptive-cache-max-hits","16")]:
            for args in [(flag,value,"--adaptive-cache","off"),("--adaptive-cache","off",flag,value)]:
                r=self.run_cli(*args);self.assertIn("require an enabled",r.stderr)
            for args in [(flag,value,"--adaptive-cache","conservative"),("--adaptive-cache","conservative",flag,value)]:
                r=self.run_cli(*args);self.assertNotIn("require an enabled",r.stderr);self.assertNotIn("must be",r.stderr)
                self.assertNotIn("test evaluation budget exceeded",r.stderr)
            r=subprocess.run(["bin/h3cli","--resume-sampler-state","/missing-controls-state",flag,value],capture_output=True,text=True)
            self.assertIn("cannot open state",r.stderr)
            for decode in ("--decode-av-state","--decode-still-latent"):
                r=subprocess.run(["bin/h3cli",decode,"/missing-controls-state",flag,value],capture_output=True,text=True)
                self.assertIn("decode-only",r.stderr)

    def test_help(self):
        r = subprocess.run(["bin/h3cli", "--help"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn("--adaptive-cache", r.stdout + r.stderr)
        self.assertIn("--subblock-sparsity", r.stdout + r.stderr)
        self.assertIn("--adaptive-cache-warmup", r.stdout + r.stderr)
        self.assertIn("--subblock-warmup", r.stdout + r.stderr)

    def test_warmup_ranges_and_dependencies(self):
        for flag,feature in [("--adaptive-cache-warmup",("--adaptive-cache","conservative")),
                             ("--subblock-warmup",("--cuda-attention","subblock"))]:
            for value in ["-1","0","1","17","2.5","nan","2x","99999999999999999999"]:
                with self.subTest(flag=flag,value=value):
                    r=self.run_cli(*feature,flag,value)
                    self.assertNotEqual(r.returncode,0);self.assertIn(flag[2:],r.stderr)
            r=self.run_cli(flag,"2")
            self.assertNotEqual(r.returncode,0);self.assertIn("requires",r.stderr)
            for value in [2,4,10,16]:
                for args in [(*feature,flag,str(value),"--steps",str(value+1)),
                             ("--steps",str(value+1),flag,str(value),*feature)]:
                    r=self.run_cli(*args)
                    self.assertNotEqual(r.returncode,0);self.assertIn("at least two steps",r.stderr)
                # Reaches backend/model checks, with exactly two remaining steps.
                r=self.run_cli(flag,str(value),"--steps",str(value+2),*feature)
                self.assertNotEqual(r.returncode,0)
                self.assertNotIn("warmup",r.stderr);self.assertNotIn("unrecognized option",r.stderr)


if __name__ == "__main__":
    unittest.main()
