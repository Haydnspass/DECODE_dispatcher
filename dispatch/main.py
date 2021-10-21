import subprocess
import psutil
import os
from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

watch_folder = "/mnt/t2ries/DECODE"
decode_default = "/home/riesgroup/xconda3/envs/decode_v010/bin/python"
working_folder = "/home/riesgroup/git/decode"


@app.get("/status")
async def status() -> Dict[str, str]:
    return {
        "health": "Dispatcher alive.",
        "watch_folder": watch_folder,
    }


@app.get("/status_gpus")
async def status_gpu() -> Dict[str, str]:
    return {"cuda:0": "100% util, 10GB mem"}


@app.get("/envs")
async def envs() -> List[str]:
    decode_envs = ["/home/AD/muellelu/xconda3/bin/python"]
    return decode_envs


@app.post("/submit_training")
async def submit_training(path: str) -> int:
    p = subprocess.Popen([
        decode_default,
        "-m", "decode.neuralfitter.train.live_engine",
        "-p", f"{watch_folder}/{path}",
    ], cwd=working_folder)
    pid = p.pid
    return pid


class Fit(BaseModel):
    frame_path: str
    frame_meta_path: str
    model_path: str
    param_path: str
    emitter_path: str
    device: str


@app.post("/submit_fit")
async def submit_fit(fit: Fit) -> int:
    p = subprocess.Popen([
        decode_default,
        "-m", "decode.neuralfitter.inference.inference",
        "--frame_path", f"{watch_folder}/{fit.frame_path}",
        "--frame_meta_path", f"{watch_folder}/{fit.frame_meta_path}",
        "--model_path", f"{watch_folder}/{fit.model_path}",
        "--param_path", f"{watch_folder}/{fit.param_path}",
        "--emitter_path", f"{watch_folder}/{fit.emitter_path}",
        "--device", f"{fit.device}",
    ], cwd=working_folder)
    pid = p.pid
    return pid


@app.post("/kill")
async def kill(pid: int):
    p = psutil.Process(pid)
    p.terminate()
