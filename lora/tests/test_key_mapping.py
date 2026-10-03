from common import *


class Mapping(Fixture):
    def test_native_peft_prefixes(self):
        for prefix in ['', 'diffusion_model.', 'model.diffusion_model.', 'transformer.', 'base_model.model.', 'base_model.model.transformer.']:
            with self.subTest(prefix=prefix):
                self.adapter_file(prefix+'blocks.0.attn.qkv_proj')
                targets, _, _ = f.adapter_targets(self.adapter, 1, self.cp)
                self.assertEqual((targets[0].tensor, targets[0].start, targets[0].stop), (QKV, 0, 6))

    def test_comfy_up_down_and_alpha(self):
        prefix = 'lora_unet_blocks_0_attn_qkv_proj'
        save(self.adapter, {prefix+'.lora_down.weight':self.a, prefix+'.lora_up.weight':self.b,
                            prefix+'.alpha':np.array(4, dtype='f')})
        targets, _, _ = f.adapter_targets(self.adapter, .5, self.cp)
        self.assertEqual(targets[0].scale, 1)
        self.assertEqual(targets[0].convention, 'comfyui')

    def test_separate_qkv(self):
        tensors = {}
        for i, part in enumerate(('to_q', 'to_k', 'to_v')):
            p = 'base_model.model.transformer.transformer_blocks.0.attn.'+part
            tensors[p+'.lora_A.default.weight'] = self.a
            tensors[p+'.lora_B.default.weight'] = self.b[i*2:i*2+2]
        save(self.adapter, tensors)
        self.run_fold()
        actual = f.Checkpoint(self.out).tensors[QKV].read()
        expected = f.round_bf16(self.w+self.b@self.a)
        np.testing.assert_array_equal(reference_bf16(actual), expected)

    def test_diffusers_out(self):
        self.a, self.b = self.a[:, :2], self.b[:4]
        self.adapter_file('transformer.transformer_blocks.0.attn.to_out.0')
        targets, _, _ = f.adapter_targets(self.adapter, 1, self.cp)
        self.assertEqual(targets[0].tensor, OUT)

    def test_unknown_and_extras(self):
        for tensors in [
            {'unknown.lora_A.weight':self.a, 'unknown.lora_B.weight':self.b},
            {'blocks.0.attn.qkv_proj.lora_A.weight':self.a},
            {'blocks.0.attn.qkv_proj.lora_B.weight':self.b},
            {'other.weight':self.a},
        ]:
            save(self.adapter, tensors)
            with self.assertRaises(f.FoldError): f.adapter_targets(self.adapter, 1, self.cp)

    def test_architectural_rejection(self):
        for extra in ['pdd_heads.0.weight', 'vdn.weight', 'controlnet.weight', 'lora_magnitude_vector']:
            self.adapter_file(extras={extra:self.a})
            with self.assertRaisesRegex(f.FoldError, 'PDD requires special output heads'):
                f.adapter_targets(self.adapter, 1, self.cp)
        self.adapter_file(metadata={'adapter_config':json.dumps({'use_rslora':True})})
        with self.assertRaises(f.FoldError): f.adapter_targets(self.adapter, 1, self.cp)

    def test_duplicate_alias_and_overlap(self):
        p = 'blocks.0.attn.qkv_proj'
        for extras in [
            {p+'.lora_down.weight':self.a},
            {'diffusion_model.'+p+'.lora_A.weight':self.a,'diffusion_model.'+p+'.lora_B.weight':self.b},
            {'blocks.0.attn.to_q.lora_A.weight':self.a,'blocks.0.attn.to_q.lora_B.weight':self.b[:2]},
        ]:
            self.adapter_file(extras=extras)
            with self.assertRaisesRegex(f.FoldError, 'duplicate'): f.adapter_targets(self.adapter, 1, self.cp)

    def test_shapes(self):
        for a, b in [(self.a.T, self.b), (self.a, self.b.T), (self.a[:1], self.b), (self.a, self.b[:3])]:
            save(self.adapter, {'blocks.0.attn.qkv_proj.lora_A.weight':a, 'blocks.0.attn.qkv_proj.lora_B.weight':b})
            with self.assertRaisesRegex(f.FoldError, 'shape mismatch'): f.adapter_targets(self.adapter, 1, self.cp)

    def test_metadata(self):
        for metadata in [{'lora_rank':'3'}, {'alpha':'nan'}, {'alpha':'2','lora_alpha':'4'},
                         {'alpha_pattern':'{}'}, {'adapter_config':'{"r":3}'}, {'alpha':'-1'},
                         {'adapter_config':'{"fan_in_fan_out":true}'}, {'peft_type':'ADALORA'}]:
            self.adapter_file(metadata=metadata)
            with self.assertRaises(f.FoldError): f.adapter_targets(self.adapter, 1, self.cp)
        self.adapter_file(metadata={'lora_alpha':'4','lora_rank':'2'})
        targets, _, _ = f.adapter_targets(self.adapter, .25, self.cp)
        self.assertEqual(targets[0].scale, .5)

    def test_ordinary_adapter_description_is_not_architecture(self):
        self.adapter_file(metadata={'description':'Dora in an adorable portrait'})
        targets, record, _ = f.adapter_targets(self.adapter, 1, self.cp)
        self.assertEqual(len(targets), 1)
        self.assertIsNone(record['profile'])

    def test_scalar_alpha_types(self):
        for dtype, numpy_dtype in [('F64', '<f8'), ('I64', '<i8'), ('F16', '<f2')]:
            self.adapter_file(extras={'blocks.0.attn.qkv_proj.alpha':(dtype,np.array(4,dtype=numpy_dtype))})
            targets, _, _ = f.adapter_targets(self.adapter, .25, self.cp)
            self.assertEqual(targets[0].scale, .5)

    def test_profiles_are_not_filenames(self):
        p = self.root/'minimax_h3_turbo_v4_step600_ema.safetensors'
        self.adapter_file(path=p)
        _, record, _ = f.adapter_targets(p, 1, self.cp)
        self.assertIsNone(record['profile'])
        _, record, _ = f.adapter_targets(p, 1, self.cp, explicit_profile='larryvrh-turbo-v4-step600-ema')
        self.assertEqual(record['profile_source'], 'explicit')

    def test_scale_punctuation(self):
        p = self.root/'a:b [test]-0.1.safetensors'
        self.adapter_file(path=p)
        self.assertEqual(f.parse_lora(str(p)), (p, 1))
        for scale in ('0', '.7', '1', '1.2', '-1', '1e-2'):
            self.assertEqual(f.parse_lora(str(p)+':'+scale), (p, float(scale)))
        for scale in ('', 'bad', 'nan', 'inf', '1e90'):
            with self.assertRaises(f.FoldError): f.parse_lora(str(p)+':'+scale)
