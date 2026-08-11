from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_seedvr2_rocm_workflow_preserves_video_and_uses_bounded_uhd_profile():
    workflow = json.loads(
        (ROOT / "workflows" / "seedvr2-4k-video-upscale-rocm.json").read_text()
    )
    nodes = {node["id"]: node for node in workflow["nodes"]}

    assert nodes[21]["type"] == "LoadVideo"
    assert nodes[22]["type"] == "GetVideoComponents"
    assert nodes[24]["type"] == "CreateVideo"
    assert nodes[23]["type"] == "SaveVideo"

    upscaler = nodes[10]
    assert upscaler["type"] == "SeedVR2VideoUpscaler"
    assert upscaler["widgets_values"] == [
        42,
        "fixed",
        2160,
        3840,
        5,
        True,
        "lab",
        3,
        0,
        0,
        0,
        "cpu",
        True,
    ]
    assert (upscaler["widgets_values"][4] - 1) % 4 == 0

    assert nodes[14]["widgets_values"] == [
        "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "cuda:0",
        16,
        False,
        "cpu",
        False,
        "sdpa",
    ]
    assert nodes[13]["widgets_values"] == [
        "ema_vae_fp16.safetensors",
        "cuda:0",
        True,
        768,
        128,
        True,
        768,
        128,
        "false",
        "cpu",
        False,
    ]


def test_seedvr2_models_have_one_durable_authority():
    model_paths = (ROOT / "containers" / "extra_model_paths.yaml").read_text()
    state_recipe = (ROOT / "justfile").read_text()

    assert "seedvr2: SEEDVR2" in model_paths
    assert 'model_dir="{{state}}/models/SEEDVR2"' in state_recipe
    assert "seedvr2_ema_3b_fp8_e4m3fn.safetensors" in state_recipe
    assert "ema_vae_fp16.safetensors" in state_recipe
