import os
import tempfile
from pathlib import Path

import pandas as pd
import torch
from decord import VideoReader
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from dataset.videoLoader import get_selected_indexs, pad_index
from utils.misc import load_config
from utils.utils import load_model
from utils.video_augmentation import Compose, Normalize, PermuteImage, Resize, ToFloatTensor


class PredictResponse(BaseModel):
    top_k: list


def build_eval_transform(cfg):
    return Compose(
        Resize(cfg["data"]["vid_transform"]["IMAGE_SIZE"]),
        ToFloatTensor(),
        PermuteImage(),
        Normalize(
            cfg["data"]["vid_transform"]["NORM_MEAN_IMGNET"],
            cfg["data"]["vid_transform"]["NORM_STD_IMGNET"],
        ),
    )


def load_label_map(path):
    label_map = {}
    lut_path = Path(path)
    if not lut_path.exists():
        return label_map

    df = pd.read_csv(lut_path)
    if {"id_label_in_documents", "name"}.issubset(df.columns):
        for _, row in df.iterrows():
            label_map[int(row["id_label_in_documents"])] = str(row["name"])
    return label_map


class InferenceService:
    def __init__(self, cfg_path, lookup_table_path=None, device="cpu"):
        self.cfg = load_config(cfg_path)
        self.device = torch.device(device)
        self.model = load_model(self.cfg).to(self.device)
        self.model.eval()
        self.transform = build_eval_transform(self.cfg)
        self.lookup_table_path = lookup_table_path
        self.label_map = load_label_map(lookup_table_path) if lookup_table_path else {}

    def _preprocess_video(self, video_path):
        vr = VideoReader(str(video_path), width=320, height=256)
        vlen = len(vr)

        index_setting = self.cfg["data"]["transform_cfg"].get(
            "index_setting", ["segment", "pad", "segment", "pad"]
        )
        selected_index, pad = get_selected_indexs(
            vlen,
            self.cfg["data"]["num_output_frames"],
            is_train=False,
            setting=index_setting,
            temporal_stride=self.cfg["data"]["temporal_stride"],
        )

        if pad is not None:
            selected_index = pad_index(selected_index, pad).tolist()

        frames = vr.get_batch(selected_index).asnumpy()
        clip = [self.transform(frame) for frame in frames]
        clip = torch.stack(clip, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
        return clip.to(self.device)

    def predict(self, video_path, top_k=5):
        clip = self._preprocess_video(video_path)

        with torch.no_grad():
            if self.cfg["data"].get("model_name") == "UsimKD":
                outputs = self.model(rgb_center=clip)
            else:
                outputs = self.model(clip=clip)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            k = min(top_k, probs.shape[-1])
            top_probs, top_indices = torch.topk(probs[0], k=k)

        results = []
        for rank, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist()), start=1):
            label_name = self.label_map.get(idx + 1, self.label_map.get(idx, "N/A"))
            results.append(
                {
                    "rank": rank,
                    "class_id": idx,
                    "probability": float(prob),
                    "label": label_name,
                }
            )
        return results


MODEL_CONFIG = os.getenv("MODEL_CONFIG", "configs/deploy/UFOneView_MultiVSL200_server.yaml")
LOOKUP_TABLE = os.getenv("LOOKUP_TABLE", "data/MultiVSL200/lookuptable.csv")
DEVICE = os.getenv("MODEL_DEVICE", "cpu")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

service = InferenceService(MODEL_CONFIG, LOOKUP_TABLE, DEVICE)

app = FastAPI(title="M3-SLR Inference API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(service.device)}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), top_k: int = DEFAULT_TOP_K):
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        predictions = service.predict(tmp_path, top_k=top_k)
        return PredictResponse(top_k=predictions)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
