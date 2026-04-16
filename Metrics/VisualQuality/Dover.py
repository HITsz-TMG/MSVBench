import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
DOVER_TOOL_DIR = os.path.join(BASE_DIR, "VisualQuality", "tools", "DOVER")
sys.path.append(DOVER_TOOL_DIR)
import torch
import pickle as pkl

import decord
import numpy as np
import yaml

from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
from dover.models import DOVER

mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)


def fuse_results(results: list):
    x = (results[0] - 0.1107) / 0.07355 * 0.6104 + (
        results[1] + 0.08285
    ) / 0.03774 * 0.3896
    print(x)
    return 1 / (1 + np.exp(-x))


def gaussian_rescale(pr):
    # The results should follow N(0,1)
    pr = (pr - np.mean(pr)) / np.std(pr)
    return pr


def uniform_rescale(pr):
    # The result scores should follow U(0,1)
    return np.arange(len(pr))[np.argsort(pr).argsort()] / len(pr)


def rescale_results(results: list, vname="undefined"):
    dbs = {
        "livevqc": "LIVE_VQC",
        "kv1k": "KoNViD-1k",
        "ltest": "LSVQ_Test",
        "l1080p": "LSVQ_1080P",
        "ytugc": "YouTube_UGC",
    }
    for abbr, full_name in dbs.items():
        pred_path = os.path.join(DOVER_TOOL_DIR, "dover_predictions", f"val-{abbr}.pkl")
        with open(pred_path, "rb") as f:
            pr_labels = pkl.load(f)
        aqe_score_set = pr_labels["resize"]
        tqe_score_set = pr_labels["fragments"]
        tqe_score_set_p = np.concatenate((np.array([results[0]]), tqe_score_set), 0)
        aqe_score_set_p = np.concatenate((np.array([results[1]]), aqe_score_set), 0)
        tqe_nscore = gaussian_rescale(tqe_score_set_p)[0]
        tqe_uscore = uniform_rescale(tqe_score_set_p)[0]
        print(f"Compared with all videos in the {full_name} dataset:")
        print(
            f"-- the technical quality of video [{vname}] is better than {int(tqe_uscore*100)}% of videos, with normalized score {tqe_nscore:.2f}."
        )
        aqe_nscore = gaussian_rescale(aqe_score_set_p)[0]
        aqe_uscore = uniform_rescale(aqe_score_set_p)[0]
        print(
            f"-- the aesthetic quality of video [{vname}] is better than {int(aqe_uscore*100)}% of videos, with normalized score {aqe_nscore:.2f}."
        )


class DOVEREvaluator:
    def __init__(self, opt_path=None, device="cuda"):
        self.device = device

        if opt_path is None:
            opt_path = os.path.join(DOVER_TOOL_DIR, "dover.yml")
        with open(opt_path, "r") as f:
            self.opt = yaml.safe_load(f)
        
        # Load DOVER model
        self.evaluator = DOVER(**self.opt["model"]["args"]).to(self.device)
        self.evaluator.load_state_dict(
            torch.load(self.opt["test_load_path"], map_location=self.device)
        )
        
        # Setup data options
        self.dopt = self.opt["data"]["val-l1080p"]["args"]
        
        # Setup temporal samplers
        self.temporal_samplers = {}
        for stype, sopt in self.dopt["sample_types"].items():
            if "t_frag" not in sopt:
                # resized temporal sampling for TQE in DOVER
                self.temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                # temporal sampling for AQE in DOVER
                self.temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )
    
    def evaluate(self, video_path):
        # View Decomposition
        views, _ = spatial_temporal_view_decomposition(
            video_path, self.dopt["sample_types"], self.temporal_samplers
        )

        for k, v in views.items():
            num_clips = self.dopt["sample_types"][k].get("num_clips", 1)
            views[k] = (
                ((v.permute(1, 2, 3, 0) - mean) / std)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                .to(self.device)
            )

        results = [r.mean().item() for r in self.evaluator(views)]
        
        # Return fused overall score
        return fuse_results(results)


if __name__ == "__main__":
    # Example usage
    evaluator = DOVEREvaluator()
    video_path = "path/to/your/video.mp4"  # Replace with an actual video path.
    score = evaluator.evaluate(video_path)
    print("Normalized fused overall score (scale in [0,1]):", score)
