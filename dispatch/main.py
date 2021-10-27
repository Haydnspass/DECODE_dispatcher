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
git_dir = Path("/home/riesgroup/git/decode")

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


env_vars = os.environ.copy()
env_vars["PYTHONPATH"] = str(git_dir)

@app.post("/submit_training", tags=["submit"])
async def submit_training(path_param: Path) -> int:

    p = subprocess.Popen([
        decode_default,
        "-m", "decode.neuralfitter.train.live_engine",
        "-p", f"{str(watch_dir)}/{str(path_param)}",
        "-l", f"{str(log_dir)}",
        ], cwd=watch_dir, env=env_vars)

    pid = p.pid
    pid_pool.add(pid)
    return pid

@app.post("/submit_fit", tags=["submit"])
async def submit_fit(path_fit_meta: Path) -> int:
    p = subprocess.Popen([
        decode_default,
        "-m", "decode.neuralfitter.inference.inference",
        "--fit_meta_path", f"{watch_dir}/{path_fit_meta}",
    ], cwd=watch_dir, env=env_vars)

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
