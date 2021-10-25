import subprocess
import psutil
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import GPUtil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()

decode_default = "/home/riesgroup/xconda3/envs/decode_dev/bin/python"

watch_dir = Path("/mnt/t2ries/decode")
log_dir = Path("/mnt/t2ries/decode/log")
working_dir = "/home/riesgroup/git/decode"

pid_pool = set()


@app.get("/status")
async def status() -> Dict[str, str]:
    return {
        "health": "Dispatcher alive.",
        "watch_dir": str(watch_dir),
        "log_dir": str(log_dir),
        "default_python_env": decode_default,
    }


@app.get("/status_gpus")
async def status_gpu() -> Dict[str, str]:
    n = torch.cuda.device_count()

    out = dict()
    for ix in range(n):
        id_gpu = f"cuda:{ix}"
        out.update({
            id_gpu: {
                "device_name": torch.cuda.get_device_name(id_gpu),
                "memory_util": GPUtil.getGPUs()[ix].memoryUsed,
                "memory_total": GPUtil.getGPUs()[ix].memoryTotal,
                "load": GPUtil.getGPUs()[ix].load,
                "temperature": GPUtil.getGPUs()[ix].temperature,
            }
        })
    return out


@app.get("/envs")
async def envs() -> List[str]:
    decode_envs = [decode_default]
    return decode_envs


@app.post("/submit_training")
async def submit_training(path_param: Path) -> int:

    p = subprocess.Popen([
        decode_default,
        "-m", "decode.neuralfitter.train.live_engine",
        "-p", f"{str(watch_dir)}/{str(path_param)}",
        "-l", f"{str(log_dir)}",
        ], cwd=working_dir)

    pid = p.pid
    pid_pool.add(pid)
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
        "--frame_path", f"{watch_dir}/{fit.frame_path}",
        "--frame_meta_path", f"{watch_dir}/{fit.frame_meta_path}",
        "--model_path", f"{watch_dir}/{fit.model_path}",
        "--param_path", f"{watch_dir}/{fit.param_path}",
        "--emitter_path", f"{watch_dir}/{fit.emitter_path}",
        "--device", f"{fit.device}",
    ], cwd=working_dir)

    pid = p.pid
    pid_pool.add(pid)
    return pid


@app.post("/kill")
async def kill(pid: int):
    if not pid in pid_pool:
        raise HTTPException(status_code=400, detail="Can not kill process that is not a training.")

    p = psutil.Process(pid)
    p.terminate()

    pid_pool.remove(pid)
