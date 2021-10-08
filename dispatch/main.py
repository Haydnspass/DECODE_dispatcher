import subprocess
import psutil
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import GPUtil
from fastapi import FastAPI, HTTPException


app = FastAPI()

decode_default = "/home/riesgroup/xconda3/envs/decode_v010/bin/python"

watch_dir = Path("/mnt/t2ries/decode/experiments")
log_dir = Path("/mnt/t2ries/decode/log")
pid_pool = set()


@app.get("/status")
async def status() -> Dict[str, str]:
    return {
        "health": "Dispatcher alive.",
        "watch_dir": str(watch_dir),
        "log_dir": str(log_dir),
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


@app.post("/submit")
async def submit(path_param: Path) -> int:

    p = subprocess.Popen([
        decode_default,
        "-m",
        "decode.neuralfitter.train.live_engine",
        "-p",
        f"{str(watch_dir)}/{str(path_param)}",
        "-l",
        f"{str(log_dir)}",
        ])
    pid = p.pid

    # bookkeeping
    pid_pool.add(pid)

    return pid


@app.post("/kill")
async def kill(pid: int):
    if not pid in pid_pool:
        raise HTTPException(status_code=400, detail="Can not kill process that is not a training.")

    p = psutil.Process(pid)
    p.terminate()

    pid_pool.remove(pid)
