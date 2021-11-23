import subprocess
import psutil
import os
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional

import torch
import GPUtil
from fastapi import FastAPI, HTTPException


app = FastAPI()

decode_default = "/home/riesgroup/xconda3/envs/decode_dev/bin/python"

watch_dir = Path("/mnt/t2ries/decode")
log_dir = Path("/mnt/t2ries/decode/log")
git_dir = Path("/home/riesgroup/git/decode")

pid_pool = {"training": set(), "fit": set()}


@app.get("/status", tags=["status"])
async def status() -> Dict[str, str]:
    return {
        "health": "Dispatcher alive.",
        "watch_dir": str(watch_dir),
        "log_dir": str(log_dir),
        "default_python_env": decode_default,
    }


@app.get("/status_gpus", tags=["status"])
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
    log_file = watch_dir / path_param.parent / "out.log"
    f = open(log_file, "w+")

    p = subprocess.Popen([
        decode_default,
        "-m", "decode.neuralfitter.train.live_engine",
        "-p", f"{str(watch_dir)}/{str(path_param)}",
        "-l", f"{str(log_dir)}",
        ], cwd=watch_dir, env=env_vars, stdout=f, stderr=f, close_fds=True)

    pid = p.pid
    pid_pool["training"].add(pid)
    return pid

@app.post("/submit_fit", tags=["submit"])
async def submit_fit(path_fit_meta: Path) -> int:
    log_file = watch_dir / path_fit_meta.parent / "out.log"
    f = open(log_file, "w+")

    p = subprocess.Popen([
        decode_default,
        "-m", "decode.neuralfitter.inference.inference",
        "--fit_meta_path", f"{watch_dir}/{path_fit_meta}",
    ], cwd=watch_dir, env=env_vars, stdout=f, stderr=f, close_fds=True)

    pid = p.pid
    pid_pool["fit"].add(pid)
    return pid


@app.get("/status_processes", tags=["status"])
async def status_proc() -> Dict[str, Dict[int, str]]:
    pid_stat_all = {p.info['pid']: p.info['status'] for p in psutil.process_iter(['pid', 'status'])}

    pid_stat = dict.fromkeys(pid_pool.keys())

    for p_type in pid_stat:
        pid_stat[p_type] = dict.fromkeys(pid_pool[p_type])

        for p in pid_stat[p_type]:
            if p in pid_stat_all:
                pid_stat[p_type][p] = pid_stat_all[p]
            else:
                pid_stat[p_type][p] = "not found"

    return pid_stat


@app.post("/kill")
async def kill(pid: int):
    if not pid in list(chain(pid_pool.values())):
        raise HTTPException(status_code=400, detail="Can not kill process that is not a training or fit.")

    p = psutil.Process(pid)
    p.terminate()

    pid_pool.remove(pid)
