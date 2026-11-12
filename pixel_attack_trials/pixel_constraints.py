"""Image-space constraints for targeted VLM answer optimization.

These are new adapters, not implementations of Agent Smith, Sparse-RS or TnT.
Low-frequency optimization uses a truncated orthonormal DCT parameterization;
clipping and PNG quantization can introduce high frequencies (reported below).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULTS = dict(update_rule="legacy", constraint="global", border_width=16, frequency_ratio=0.125,
                sparse_pixels=256, patch_width=112, patch_height=112,
                patch_x=None, patch_y=0, flower_petals=5,
                flower_image=None, mask_image=None)


def add_arguments(parser):
    parser.add_argument("--update-rule", choices=("legacy", "revised"), default="legacy")
    parser.add_argument("--constraint", choices=("global", "border", "lowfreq", "flower", "sparse"), default="global")
    parser.add_argument("--border-width", type=int, default=16, help="Border width AFTER model resizing")
    parser.add_argument("--frequency-ratio", type=float, default=0.125, help="DCT retained fraction per axis; clipping/PNG may leak high frequencies")
    parser.add_argument("--sparse-pixels", type=int, default=256, help="L0 budget counts RGB pixel locations, not channels")
    parser.add_argument("--patch-width", type=int, default=112)
    parser.add_argument("--patch-height", type=int, default=112)
    parser.add_argument("--patch-x", type=int, default=None, help="Default right aligned, AFTER model resizing")
    parser.add_argument("--patch-y", type=int, default=0)
    parser.add_argument("--flower-petals", type=int, default=5)
    parser.add_argument("--flower-image", type=Path, help="Optional flower reference, preferably transparent RGBA; epsilon is measured from this reference")
    parser.add_argument("--mask-image", type=Path, help="Optional binary silhouette; white/alpha >=128 is editable")


def validate(args):
    if args.border_width < 1 or args.sparse_pixels < 1:
        raise ValueError("border-width/sparse-pixels must be positive")
    if not math.isfinite(args.frequency_ratio) or not 0 < args.frequency_ratio < 1:
        raise ValueError("frequency-ratio must be strictly between 0 and 1")
    if min(args.patch_width, args.patch_height, args.flower_petals) < 1:
        raise ValueError("Patch size and petal count must be positive")
    if args.patch_y < 0 or (args.patch_x is not None and args.patch_x < 0):
        raise ValueError("Patch coordinates must be nonnegative")
    for name in ("flower_image", "mask_image"):
        value = getattr(args, name)
        if value is not None:
            if args.constraint != "flower":
                raise ValueError(f"--{name.replace('_', '-')} requires --constraint flower")
            with Image.open(value) as source:
                source.verify()


def dct_basis(size, retained, device, torch):
    frequency = torch.arange(retained, device=device, dtype=torch.float32)[:, None]
    location = torch.arange(size, device=device, dtype=torch.float32)[None, :]
    basis = torch.cos(math.pi / size * (location + 0.5) * frequency) * math.sqrt(2 / size)
    basis[0] /= math.sqrt(2)
    return basis


class PixelConstraint:
    def __init__(self, args, clean, torch):
        self.args, self.clean, self.torch = args, clean, torch
        self.kind, self.epsilon = args.constraint, args.epsilon / 255
        self.anchor = clean.detach().clone()
        self.mask = torch.ones_like(clean[:, :1])
        H, W = clean.shape[-2:]
        self.info = {"implementation_version": 1, "kind": self.kind}
        if getattr(args, "update_rule", "legacy") == "revised":
            self.info.update(implementation_version=2, update_rule="revised")
        if self.kind == "border":
            b = args.border_width
            if 2 * b >= min(H, W):
                raise ValueError("Border consumes the image; reduce --border-width")
            self.mask[:, :, b:-b, b:-b] = 0
            self.info.update(border_width=b)
        elif self.kind == "flower":
            w, h = args.patch_width, args.patch_height
            x, y = W - w if args.patch_x is None else args.patch_x, args.patch_y
            if x < 0 or x + w > W or y + h > H:
                raise ValueError(f"Flower patch {w}x{h} at {x},{y} is outside resized image {W}x{H}")
            yy, xx = np.mgrid[:h, :w]
            xx, yy = (xx + .5 - w / 2) / (w / 2), (yy + .5 - h / 2) / (h / 2)
            radius = .68 + .28 * np.cos(args.flower_petals * np.arctan2(yy, xx))
            silhouette = np.hypot(xx, yy) <= radius
            reference = None
            if args.flower_image:
                with Image.open(args.flower_image) as source:
                    has_alpha = "A" in source.getbands() or "transparency" in source.info
                    rgba = source.convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)
                    values = np.array(rgba)
                    if has_alpha:
                        silhouette = values[:, :, 3] >= 128
                    reference = torch.tensor(values[:, :, :3].copy(), device=clean.device,
                                             dtype=clean.dtype).permute(2, 0, 1)[None] / 255
                self.info.update(flower_image_sha256=hashlib.sha256(Path(args.flower_image).read_bytes()).hexdigest())
            if args.mask_image:
                with Image.open(args.mask_image) as source:
                    channel = source.convert("RGBA").getchannel("A") if "A" in source.getbands() else source.convert("L")
                    silhouette = np.array(channel.resize((w, h), Image.Resampling.NEAREST)) >= 128
                self.info.update(mask_image_sha256=hashlib.sha256(Path(args.mask_image).read_bytes()).hexdigest())
            if not silhouette.any():
                raise ValueError("Flower mask is empty")
            local_mask = torch.tensor(silhouette, device=clean.device, dtype=clean.dtype)[None, None]
            self.mask.zero_()
            self.mask[:, :, y:y+h, x:x+w] = local_mask
            if reference is not None:
                region = clean[:, :, y:y+h, x:x+w]
                self.anchor[:, :, y:y+h, x:x+w] = region * (1-local_mask) + reference * local_mask
            self.info.update(patch_box=[x, y, w, h], petals=args.flower_petals,
                             reference_flower=reference is not None,
                             appearance_claim="Reference-bounded pixels only; no perceptual naturalness guarantee")
        elif self.kind == "lowfreq":
            kh, kw = max(1, math.ceil(H * args.frequency_ratio)), max(1, math.ceil(W * args.frequency_ratio))
            self.bh = dct_basis(H, kh, clean.device, torch)
            self.bw = dct_basis(W, kw, clean.device, torch)
            self.info.update(frequency_ratio=args.frequency_ratio, coefficient_shape=[1, 3, kh, kw],
                             frequency_claim="Strictly truncated latent DCT; clipped/quantized image may leak high frequencies")
        elif self.kind == "sparse":
            if args.sparse_pixels >= H * W:
                raise ValueError("Sparse pixel budget must be smaller than the resized image area")
            self.info.update(sparse_pixels=args.sparse_pixels, support="RGB locations; iterative hard thresholding")

    def field(self, parameter):
        if self.kind == "lowfreq":
            return self.bh.T @ parameter @ self.bw
        return parameter * self.mask

    def render(self, parameter):
        return (self.anchor + self.field(parameter)).clamp(0, 1)

    def initialize(self, generator):
        shape = self.info.get("coefficient_shape", self.clean.shape)
        parameter = self.torch.empty(shape, device=self.clean.device, dtype=self.clean.dtype)
        parameter.uniform_(-self.epsilon, self.epsilon, generator=generator)
        if self.kind == "sparse":
            # Let the first target-loss gradient choose the support, rather than
            # locking the budget into arbitrary near-epsilon random locations.
            parameter.zero_()
        with self.torch.no_grad():
            self.project_(parameter)
        return parameter

    def project_(self, parameter):
        if self.kind == "lowfreq":
            peak = self.field(parameter).abs().max().clamp_min(1e-12)
            parameter.mul_((self.epsilon / peak).clamp(max=1))
        else:
            parameter.clamp_(-self.epsilon, self.epsilon)
            parameter.copy_(((self.anchor + parameter).clamp(0, 1) - self.anchor) * self.mask)
            if self.kind == "sparse":
                # Exact L2 projection onto at most k RGB locations, after the box projection.
                scores = parameter.square().sum(dim=1).flatten()
                keep = scores.topk(self.args.sparse_pixels).indices
                support = self.torch.zeros_like(scores)
                support[keep] = 1
                parameter.mul_(support.reshape(1, 1, *parameter.shape[-2:]))

    def step_(self, parameter, gradient, alpha):
        if getattr(self.args, "update_rule", "legacy") == "revised":
            if self.kind == "sparse":
                # Select locations by estimated target-loss reduction, rather
                # than rewarding previously accumulated perturbation magnitude.
                candidate = (parameter - alpha * gradient.sign()).clamp(-self.epsilon, self.epsilon)
                candidate = (self.anchor + candidate).clamp(0, 1) - self.anchor
                benefit = -(gradient * candidate).sum(dim=1).flatten()
                indices = benefit.topk(self.args.sparse_pixels).indices
                support = self.torch.zeros_like(benefit)
                support[indices] = (benefit[indices] > 0).to(support.dtype)
                parameter.copy_(candidate * support.reshape(1, 1, *parameter.shape[-2:]))
                self.project_(parameter)
                return
            if self.kind == "lowfreq":
                # Preserve relative coefficient gradients; normalize the actual
                # spatial update instead of taking signs in coefficient space.
                direction = gradient / self.field(gradient).abs().max().clamp_min(1e-12)
                parameter.sub_(alpha * direction)
                self.project_(parameter)
                return
        direction = gradient.sign()
        if self.kind == "sparse":
            # Retain gradient magnitudes for hard-threshold support selection.
            direction = gradient / gradient.abs().max().clamp_min(1e-12)
        if self.kind == "lowfreq":
            # Normalize the spatial step, since coefficient and pixel units differ.
            direction = direction / self.field(direction).abs().max().clamp_min(1e-12)
        parameter.sub_(alpha * direction)
        self.project_(parameter)

    def validate_parameter(self, parameter):
        torch = self.torch
        expected = tuple(self.info.get("coefficient_shape", self.clean.shape))
        if tuple(parameter.shape) != expected or not torch.isfinite(parameter).all().item():
            raise ValueError("Invalid constrained parameter shape or values")
        projected = parameter.clone()
        self.project_(projected)
        if not torch.allclose(projected, parameter, atol=1e-6, rtol=0):
            raise ValueError("Checkpoint violates the constraint")

    def diagnostics(self, parameter, pixels=None):
        torch = self.torch
        image = self.render(parameter) if pixels is None else pixels
        clean, anchor = self.clean, self.anchor
        if pixels is not None:
            # Compare exported PNGs in the same integer domain; unchanged pixels
            # must not be counted as sparse modifications due to roundoff.
            clean, anchor = (v.mul(255).round().clamp(0, 255)/255 for v in (clean, anchor))
        difference = image - clean
        reference_difference = image - anchor
        threshold = 0.5 / 255 if pixels is not None else 1e-6
        changed = difference.abs().amax(dim=1) > threshold
        result = dict(changed_pixel_locations=int(changed.sum().item()),
                      max_change_from_source_255=float(difference.abs().max().item() * 255),
                      max_change_from_reference_255=float(reference_difference.abs().max().item() * 255))
        if self.kind in ("border", "flower"):
            result["outside_mask_max_change_255"] = float((difference * (1-self.mask)).abs().max().item() * 255)
        if self.kind == "lowfreq":
            retained = self.bh @ reference_difference @ self.bw.T
            projection = self.bh.T @ retained @ self.bw
            energy = reference_difference.square().sum()
            result["outside_retained_DCT_energy_fraction"] = float(((reference_difference-projection).square().sum() / energy.clamp_min(1e-20)).item())
        return result
