import subprocess
from typing import Dict, List, Optional

from fastapi import FastAPI

app = FastAPI()


@app.get("/status")
def status() -> Dict[str, str]:
    return {
        "health": "Dispatcher alive.",
        "watch_folder": "/mnt/t2ries/DECODE"
    }


@app.get("/status_gpus")
def status_gpu() -> Dict[str, str]:
    return {"cuda:0": "100% util, 10GB mem"}


@app.get("/envs")
def envs() -> List[str]:
    decode_envs = ["/home/AD/muellelu/xconda3/bin/python"]
    return decode_envs


@app.post("/submit")
def submit(path: str) -> int:
    return 42


@app.post("/kill")
def kill(pid: int):
    return
