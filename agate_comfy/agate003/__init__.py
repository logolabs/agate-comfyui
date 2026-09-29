"""Agate by LogoLabs: a 0.26B text-to-image model at 512 px (multi-resolution thinker-steered convolutional
flow model + fine-tuned Ettin-68M text encoder + SD-VAE decoder). See AgatePipeline."""
from .marking import detect_watermark, provenance, save
from .pipeline import AgatePipeline

__all__ = ["AgatePipeline", "detect_watermark", "provenance", "save"]
