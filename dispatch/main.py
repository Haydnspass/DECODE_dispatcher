import subprocess
import psutil
import os
from typing import Dict, List, Optional

from fastapi import FastAPI

app = FastAPI()

watch_folder = "/mnt/t2ries/DECODE"
decode_default = "/home/riesgroup/xconda3/envs/decode_v010/bin/python"

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


@app.post("/submit")
async def submit(path: str) -> int:
    p = subprocess.Popen([decode_default, "-m", "decode.neuralfitter.train.live_engine", "-p", f"{watch_folder}/{path}"])
    pid = p.pid
    return pid


@app.post("/kill")
async def kill(pid: int):
    p = psutil.Process(pid)
    p.terminate()
