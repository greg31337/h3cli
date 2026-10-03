from unittest.mock import patch
from common import *


class Mathematics(Fixture):
    def test_single_scales(self):
        self.adapter_file()
        for scale in (0, .25, 1, 1.5, -1):
            with self.subTest(scale=scale):
                self.run_fold([(self.adapter, scale)], overwrite=True)
                cp = f.Checkpoint(self.out)
                np.testing.assert_array_equal(reference_bf16(cp.tensors[QKV].read()), reference_bf16(self.w+np.float32(scale)*(self.b@self.a)))
                np.testing.assert_array_equal(cp.tensors[OUT].read(), self.w2)
                self.unchanged()

    def test_alpha_rank(self):
        self.adapter_file(metadata={'lora_alpha':'4','rank':'2'})
        self.run_fold([(self.adapter, .3)])
        np.testing.assert_array_equal(reference_bf16(f.Checkpoint(self.out).tensors[QKV].read()),
                                      reference_bf16(self.w+np.float32(.6)*(self.b@self.a)))

    def test_different_targets(self):
        self.adapter_file()
        p = self.root/'second.safetensors'
        save(p, {'blocks.0.attn.out_proj.lora_A.weight':self.a[:, :2], 'blocks.0.attn.out_proj.lora_B.weight':self.b[:4]})
        self.run_fold([(self.adapter, 1), (p, .7)])
        cp = f.Checkpoint(self.out)
        np.testing.assert_array_equal(reference_bf16(cp.tensors[OUT].read()), reference_bf16(self.w2+np.float32(.7)*(self.b[:4]@self.a[:, :2])))

    def test_same_target_round_once(self):
        self.w[:] = 1
        save(self.base/'model.safetensors', {QKV:bf(self.w)})
        self.a[:]=.025
        self.b[:]=.06  # Two individually sub-half-ULP updates combine above half.
        self.adapter_file()
        self.run_fold([(self.adapter, 1), (self.adapter, 1)])
        expected = reference_bf16(self.w+self.b@self.a+self.b@self.a)
        np.testing.assert_array_equal(reference_bf16(f.Checkpoint(self.out).tensors[QKV].read()), expected)
        np.testing.assert_array_equal(expected, np.full_like(expected, 0x3f81))

    def test_bf16_ties_even(self):
        values=np.array([1+2**-8,1+3*2**-8,-1-2**-8,0,1e-40], dtype='f')
        np.testing.assert_array_equal(f.round_bf16(values), reference_bf16(values))
        for value in (np.inf, np.nan, np.finfo('f').max):
            with self.assertRaises(f.FoldError): f.round_bf16(np.array([value], dtype='f'))

    def test_probe_detects_corruption(self):
        self.adapter_file()
        good = f.round_bf16
        def bad(w):
            v=good(w)
            v ^= np.uint16(0x8000)
            return v
        with patch.object(f, 'round_bf16', bad):
            with self.assertRaisesRegex(f.FoldError, 'parity failed'): self.run_fold()
        self.assertFalse(self.out.exists())
        self.unchanged()

    def test_memory_guard_and_chunk_equivalence(self):
        self.adapter_file()
        with self.assertRaisesRegex(f.FoldError, 'too small'): self.run_fold(memory_mib=.0001)
        self.assertFalse(self.out.exists())
        self.run_fold(memory_mib=.002)
        first=(self.out/'model.safetensors').read_bytes()
        self.run_fold(overwrite=True)
        self.assertEqual(first,(self.out/'model.safetensors').read_bytes())

    def test_nonfinite_adapter_failure(self):
        self.b[1, 1]=np.nan
        self.adapter_file()
        with self.assertRaisesRegex(f.FoldError, 'non-finite'): self.run_fold()
        self.assertFalse(self.out.exists())
        self.unchanged()

    def test_streamed_probe_metrics(self):
        a=np.arange(300,dtype='f').reshape(100,3)/100
        b=a+np.cos(a)/100
        stats=f.ProbeMetrics()
        for start in range(0,100,7): stats.add(a[start:start+7],b[start:start+7])
        result=stats.finish()
        x,y=a.astype('d').ravel(),b.astype('d').ravel()
        self.assertAlmostEqual(result['relative_l2'],np.linalg.norm(x-y)/np.linalg.norm(y),places=14)
        self.assertAlmostEqual(result['cosine_similarity'],np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)),places=14)
        self.assertAlmostEqual(result['mean_absolute_error'],np.mean(np.abs(x-y)),places=14)
