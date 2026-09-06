#!/usr/bin/env python3
"""
D-FINE PyTorch (.pth) to Clean ONNX Exporter (deployment_v2)

Exports the D-FINE model backbone and detection heads to ONNX without the postprocessor,
enabling direct TRT compilation with raw 'pred_logits' and 'pred_boxes' outputs.

Usage (run from D-FINE repository root):
    python3 deployment_v2/export_onnx.py \
        --config configs/dfine/dfine_hgnetv2_m_coco.yml \
        --resume models/dfine_m_obj2coco.pth \
        --output deployment_v2/models/dfine_m_obj2coco.onnx \
        --check \
        --simplify
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

# Ensure current working directory and relative paths are in sys.path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
try:
    from src.core import YAMLConfig
except ImportError:
    pass


class ExportModel(nn.Module):
    """Wraps D-FINE deploy network to return raw (pred_logits, pred_boxes)."""
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.net = model.deploy()

    def forward(self, images: torch.Tensor):
        outputs = self.net(images)
        if isinstance(outputs, dict):
            if "pred_logits" in outputs and "pred_boxes" in outputs:
                return outputs["pred_logits"], outputs["pred_boxes"]
            keys = sorted(list(outputs.keys()))
            if len(keys) >= 2:
                return outputs[keys[0]], outputs[keys[1]]
            return tuple(outputs.values())
        if isinstance(outputs, (tuple, list)):
            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs[:2])
        return outputs


def export_pth_to_onnx(config_path: str, resume_path: str, output_path: str, check: bool = True, simplify: bool = True):
    from src.core import YAMLConfig

    cfg = YAMLConfig(config_path, resume=resume_path)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if resume_path:
        print(f"Loading checkpoint weights from: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
        cfg.model.load_state_dict(state)
    else:
        print("Warning: no checkpoint provided, using randomly initialized weights...")

    model = ExportModel(cfg.model).eval()

    # Dummy input [1, 3, 640, 640]
    images = torch.randn(1, 3, 640, 640)

    if not output_path:
        output_path = resume_path.replace(".pth", ".onnx") if resume_path else "model.onnx"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Exporting model to ONNX: {output_path}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (images,),
            output_path,
            opset_version=17,
            input_names=["images"],
            output_names=["pred_logits", "pred_boxes"],
            dynamic_axes={
                "images": {0: "N"},
                "pred_logits": {0: "N"},
                "pred_boxes": {0: "N"},
            },
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

    if check:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX checker passed successfully.")

    if simplify:
        try:
            import onnx
            import onnxsim
            input_shapes = {"images": images.shape}
            onnx_model_simplify, ok = onnxsim.simplify(output_path, test_input_shapes=input_shapes)
            if ok:
                onnx.save(onnx_model_simplify, output_path)
                print("ONNX simplify completed successfully.")
        except Exception as e:
            print(f"ONNX simplify skipped/failed: {e}")

    print(f"Done. ONNX exported to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export D-FINE PyTorch checkpoint to ONNX.")
    parser.add_argument("--config", "-c", type=str, default="configs/dfine/dfine_hgnetv2_m_coco.yml", help="D-FINE YAML config")
    parser.add_argument("--resume", "-r", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", "-o", type=str, default="", help="Path for output .onnx file")
    parser.add_argument("--check", action="store_true", default=True, help="Validate exported ONNX")
    parser.add_argument("--simplify", action="store_true", default=True, help="Run onnxsim")

    args = parser.parse_args()
    export_pth_to_onnx(args.config, args.resume, args.output, args.check, args.simplify)
