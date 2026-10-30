"""Generation worker for a remote GPU (e.g. Colab): the web server sends prompts, this returns waveforms.

  WORKER_TOKEN=... python scripts/colab_worker.py --port 8000

GET  /health                  -> {"ok", "device", "loaded"}
GET  /info?model_name=...     -> loads the generator if needed -> {"sample_rate", "device"}
POST /generate                -> {"sr", "shape", "data"}: float32 (batch, 1, samples), base64

Every request needs the header X-Token: <WORKER_TOKEN> (the tunnel URL is public).
"""
import argparse
import base64
import os
import threading

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import generators as gens

app = FastAPI(title="SubEvent2Hap generation worker")
MODELS, LOCK = {}, threading.Lock()
TOKEN = os.environ.get("WORKER_TOKEN", "")
DEVICE = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def _check(token):
    if TOKEN and token != TOKEN:
        raise HTTPException(401, "bad token")


def _model(name):
    if name not in gens.GENERATORS:
        raise HTTPException(422, f"unknown generator {name}")
    with LOCK:
        if name not in MODELS:
            MODELS[name] = gens.load(name)
        return MODELS[name]


@app.get("/health")
def health(x_token: str = Header("")):
    _check(x_token)
    return {"ok": True, "device": DEVICE, "loaded": sorted(MODELS)}


@app.get("/info")
def info(model_name: str = "hapticgen", x_token: str = Header("")):
    _check(x_token)
    return {"sample_rate": _model(model_name).sample_rate, "device": DEVICE}


class GenReq(BaseModel):
    model: str
    prompts: list[str] = Field(min_length=1, max_length=32)
    duration: float = Field(gt=0, le=10)
    cfg: float = 3.0
    temperature: float = 1.0
    top_k: int = 250


@app.post("/generate")
def generate(req: GenReq, x_token: str = Header("")):
    _check(x_token)
    m = _model(req.model)
    with LOCK:
        m.set_generation_params(duration=req.duration, use_sampling=True, top_k=req.top_k, top_p=0.0,
                                temperature=req.temperature, cfg_coef=req.cfg)
        wavs = m.generate(req.prompts).detach().cpu().float().numpy()
    return {"sr": m.sample_rate, "shape": list(wavs.shape), "data": base64.b64encode(wavs.tobytes()).decode()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, host="127.0.0.1", port=ap.parse_args().port, log_level="warning")
