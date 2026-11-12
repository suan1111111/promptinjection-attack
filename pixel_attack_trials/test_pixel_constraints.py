"""CPU invariants and differentiability tests; these do not measure attack success."""
import argparse
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

import pixel_constraints as pc
import optimize_qwen_pixels as base


def options(kind, **overrides):
    values = dict(pc.DEFAULTS, epsilon=8.)
    values.update(constraint=kind, border_width=4, patch_width=24, patch_height=24,
                  sparse_pixels=17)
    values.update(overrides)
    return argparse.Namespace(**values)


class ConstraintTests(unittest.TestCase):
    def test_revised_updates_reduce_linear_objective(self):
        for kind in ('lowfreq', 'sparse'):
            clean = torch.full((1,3,56,56), .5)
            c = pc.PixelConstraint(options(kind, update_rule='revised'), clean, torch)
            p = torch.zeros(c.info.get('coefficient_shape', clean.shape))
            g = torch.randn(p.shape, generator=torch.Generator().manual_seed(17))
            c.step_(p, g, 1/255)
            c.validate_parameter(p)
            self.assertLess(float((p*g).sum()), 0)
            self.assertEqual(c.info['implementation_version'], 2)
        # A new strong location can replace the old support.
        c = pc.PixelConstraint(options('sparse', update_rule='revised', sparse_pixels=1), clean, torch)
        p = torch.zeros_like(clean)
        p[:,:,0,0] = .02
        g = torch.zeros_like(clean)
        g[:,:,10,10] = 1
        c.step_(p,g,1/255)
        self.assertEqual(float(p[:,:,0,0].abs().sum()),0)
        self.assertGreater(float(p[:,:,10,10].abs().sum()),0)

    def setUp(self):
        self.clean = torch.linspace(0, 1, 3*56*56).reshape(1, 3, 56, 56)

    def test_updates_and_checkpoint_roundtrip(self):
        for kind in ("border", "flower", "lowfreq", "sparse"):
            with self.subTest(kind=kind):
                constraint = pc.PixelConstraint(options(kind), self.clean, torch)
                parameter = constraint.initialize(torch.Generator().manual_seed(7)).requires_grad_()
                weight = torch.randn(self.clean.shape, generator=torch.Generator().manual_seed(8))
                for _ in range(3):
                    (constraint.render(parameter) * weight).sum().backward()
                    self.assertGreater(parameter.grad.abs().max().item(), 0)
                    with torch.no_grad():
                        constraint.step_(parameter, parameter.grad, 1/255)
                    parameter.grad = None
                constraint.validate_parameter(parameter.detach())
                image = constraint.render(parameter).detach()
                self.assertTrue((image >= 0).all() and (image <= 1).all())
                self.assertLessEqual((image-self.clean).abs().max().item(), 8/255+1e-6)
                if kind in ("flower", "border"):
                    self.assertEqual(((image-self.clean)*(1-constraint.mask)).abs().max().item(), 0)
                if kind == "sparse":
                    self.assertLessEqual(((image-self.clean).abs().amax(1)>1e-6).sum().item(), 17)
                buffer = io.BytesIO()
                torch.save({"delta": parameter.detach()}, buffer)
                buffer.seek(0)
                restored = torch.load(buffer, weights_only=True)["delta"]
                constraint.validate_parameter(restored)
                torch.testing.assert_close(constraint.render(restored), image)
                with torch.no_grad():
                    gradient = torch.randn_like(restored)
                    expected = parameter.detach().clone()
                    constraint.step_(expected, gradient, 1/255)
                    constraint.step_(restored, gradient, 1/255)
                    torch.testing.assert_close(expected, restored)

    def test_low_frequency_latent_and_clip_leakage(self):
        clean = torch.full_like(self.clean, .5)
        constraint = pc.PixelConstraint(options("lowfreq"), clean, torch)
        parameter = constraint.initialize(torch.Generator().manual_seed(1))
        field = constraint.field(parameter)
        projected = constraint.bh.T @ (constraint.bh @ field @ constraint.bw.T) @ constraint.bw
        torch.testing.assert_close(projected, field, atol=1e-7, rtol=1e-5)
        self.assertLess(constraint.diagnostics(parameter)["outside_retained_DCT_energy_fraction"], 1e-9)
        clipped = pc.PixelConstraint(options("lowfreq"), self.clean, torch)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            base.save_tensor_image(path, clipped.render(parameter))
            png = torch.tensor(np.array(Image.open(path))).permute(2, 0, 1)[None].float()/255
            value = clipped.diagnostics(parameter, png)["outside_retained_DCT_energy_fraction"]
            self.assertTrue(0 <= value <= 1.00001)

    def test_flower_reference_alpha_and_mask_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            pixels = np.zeros((24, 24, 4), dtype=np.uint8)
            pixels[4:20, 10:14] = [255, 80, 140, 255]
            pixels[10:14, 4:20] = [255, 80, 140, 255]
            path = Path(tmp) / "flower.png"
            Image.fromarray(pixels).save(path)
            c = pc.PixelConstraint(options("flower", flower_image=path), self.clean, torch)
            p = c.initialize(torch.Generator().manual_seed(7))
            image = c.render(p)
            self.assertGreater((c.anchor-self.clean).abs().max().item(), .1)
            self.assertLessEqual((image-c.anchor).abs().max().item(), 8/255+1e-6)
            self.assertEqual(((image-self.clean)*(1-c.mask)).abs().max().item(), 0)
            self.assertEqual(int(c.mask.sum().item()), 112)

    def test_masked_and_dct_gradients_through_original_loss_pipeline(self):
        from types import SimpleNamespace
        class ToyModel:
            def zero_grad(self, **kwargs):
                pass
            def __call__(self, pixel_values, **kwargs):
                return SimpleNamespace(loss=(pixel_values-.3).square().mean())
        examples = [dict(input_ids=torch.ones(1, 1, dtype=torch.long),
                         attention_mask=torch.ones(1, 1), labels=torch.ones(1, 1, dtype=torch.long),
                         image_grid_thw=torch.tensor([[1, 4, 4]]))]
        for kind in ("border", "flower", "sparse", "lowfreq"):
            c = pc.PixelConstraint(options(kind), self.clean, torch)
            p = c.initialize(torch.Generator().manual_seed(7)).requires_grad_()
            base.accumulate_question_gradients(ToyModel(), examples, [0], self.clean, p,
                    torch.zeros(1,3,1,1), torch.ones(1,3,1,1), torch.device("cpu"), torch, c.render)
            self.assertGreater(p.grad.abs().max().item(), 0)

    def test_reject_invalid_checkpoints_and_geometry(self):
        c = pc.PixelConstraint(options("border"), self.clean, torch)
        invalid = torch.zeros_like(self.clean)
        invalid[:, :, 20, 20] = .01
        with self.assertRaises(ValueError):
            c.validate_parameter(invalid)
        for args in (options("border", border_width=28), options("sparse", sparse_pixels=56*56),
                     options("flower", patch_width=57)):
            with self.assertRaises(ValueError):
                pc.PixelConstraint(args, self.clean, torch)


if __name__ == "__main__":
    unittest.main()
