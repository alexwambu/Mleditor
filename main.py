import os
import json
import time
import threading
import shutil
import base64
import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Form, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from transformers import pipeline
import requests
from web3 import Web3, HTTPProvider
from solcx import compile_source, install_solc, set_solc_version

# ---------- Config from ENV ----------
RPC_URL = os.getenv("RPC_URL", "http://localhost:9636")
PRIVATE_KEY = os.getenv("PRIVATE_KEY", None)
MEMORY_URL = os.getenv("MEMORY_URL", None)           # e.g. http://storage-server:8000
CHAIN_ID = int(os.getenv("CHAIN_ID", "999"))
SOLC_VERSION = os.getenv("SOLC_VERSION", "0.8.20")
DOCKER_RUN = os.getenv("DOCKER_RUN", "false").lower() in ("1","true","yes")

# ---------- Initialize ----------
w3 = Web3(HTTPProvider(RPC_URL))
app = FastAPI(title="Chatbot Builder & Ethereum Deployer")

BOTS_DIR = "bots"
os.makedirs(BOTS_DIR, exist_ok=True)
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)
OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

# Try ensure solc
try:
    install_solc(SOLC_VERSION)
    set_solc_version(SOLC_VERSION)
except Exception as e:
    print("solc installation warning:", e)

# Simple text-generation pipeline (small model)
try:
    textgen = pipeline("text-generation", model="distilgpt2")
except Exception as e:
    print("Warning: transformer pipeline load failed:", e)
    textgen = None

# ---------- Heartbeat ----------
def heartbeat():
    while True:
        try:
            print(f"[heartbeat] time={time.strftime('%Y-%m-%d %H:%M:%S')} rpc_connected={w3.is_connected()} memory={bool(MEMORY_URL)}")
        except Exception as e:
            print("[heartbeat] err", e)
        time.sleep(18)

threading.Thread(target=heartbeat, daemon=True).start()

# ---------- In-memory bot instances ----------
class BotConfig(BaseModel):
    name: str
    persona: str = "You are a helpful assistant."
    max_length: int = 128

class BotInstance:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.history = []
        self.lock = threading.Lock()

    def chat(self, prompt: str):
        with self.lock:
            self.history.append({"role":"user","text":prompt})
            if textgen is None:
                reply = "Model not available."
            else:
                pref = self.cfg.persona + "\nUser: " + prompt + "\nAssistant:"
                out = textgen(pref, max_length=self.cfg.max_length, num_return_sequences=1)
                reply = out[0]["generated_text"]
            self.history.append({"role":"bot","text":reply})
            return reply
    def get_history(self):
        return list(self.history)

bots: Dict[str, BotInstance] = {}
bots_lock = threading.Lock()

# ---------- Helpers: save to MEMORY_URL (/ml/save) ----------
def save_to_memory(filename: str, local_path: str):
    """POST file to MEMORY_URL /ml/save form field 'name' and file"""
    if not MEMORY_URL:
        return False, "MEMORY_URL not set"
    try:
        with open(local_path, "rb") as f:
            files = {"file": (filename, f)}
            data = {"name": filename}
            r = requests.post(f"{MEMORY_URL}/ml/save", data=data, files=files, timeout=20)
            return r.ok, r.text
    except Exception as e:
        return False, str(e)

# ---------- Genesis builder ----------
def build_clique_genesis(chainId:int=CHAIN_ID, signer:str=None, gasLimit:int=8000000):
    s = (signer or "0x" + "0"*40).lower().replace("0x","")
    if len(s) != 40:
        raise ValueError("signer must be 20-byte hex")
    extra = "0x" + ("00"*32) + s + ("00"*65)
    genesis = {
        "config": {
            "chainId": chainId,
            "clique": {"period": 5, "epoch": 30000},
            "homesteadBlock": 0,
            "eip150Block": 0,
            "eip155Block": 0,
            "byzantiumBlock": 0
        },
        "difficulty": "1",
        "gasLimit": hex(gasLimit),
        "alloc": {},
        "extraData": extra,
        "timestamp":"0x0"
    }
    return genesis

# ---------- ERC-20 source ----------
ERC20_SRC = r'''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
contract SimpleERC20 {
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    event Transfer(address indexed from, address indexed to, uint256 value);

    constructor(string memory _name, string memory _symbol, uint8 _decimals, uint256 _supply) {
        name = _name; symbol = _symbol; decimals = _decimals; totalSupply = _supply;
        balanceOf[msg.sender] = _supply;
        emit Transfer(address(0), msg.sender, _supply);
    }

    function transfer(address to, uint256 value) external returns (bool) {
        require(balanceOf[msg.sender] >= value, "insufficient");
        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(msg.sender, to, value);
        return true;
    }
}
'''

# ---------- Compile & deploy helpers ----------
def compile_contract(source: str, contract_name: str="SimpleERC20"):
    compiled = compile_source(source, output_values=["abi","bin"])
    key = [k for k in compiled.keys() if k.endswith(f":{contract_name}")][0]
    abi = compiled[key]["abi"]
    bin = compiled[key]["bin"]
    return abi, bin

def sign_and_send(tx, privkey, wait_receipt=True, timeout=180):
    acct = w3.eth.account.from_key(privkey)
    signed = acct.sign_transaction(tx)
    txh = w3.eth.send_raw_transaction(signed.rawTransaction)
    if not wait_receipt:
        return txh.hex()
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = w3.eth.get_transaction_receipt(txh)
            if r:
                return r
        except Exception:
            time.sleep(1)
    raise TimeoutError("tx receipt timeout")

# ---------- Worker: Provision multi-node geth cluster (generate compose) ----------
def generate_geth_node_compose(num_nodes:int=3, chain_id:int=CHAIN_ID, genesis:dict=None, base_name:str="gbtnode"):
    ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    outdir = os.path.abspath(os.path.join(OUT_DIR, f"compose_{ts}"))
    os.makedirs(outdir, exist_ok=True)

    # write genesis.json
    genesis_path = os.path.join(outdir, "genesis.json")
    gen = genesis or build_clique_genesis(chain_id, signer=None)
    with open(genesis_path, "w") as f:
        json.dump(gen, f, indent=2)

    # build docker-compose
    services = {}
    for i in range(1, num_nodes+1):
        nodename = f"{base_name}{i}"
        datadir = f"./data/{nodename}"
        os.makedirs(os.path.join(outdir, datadir), exist_ok=True)
        # simplified service using ethereum/client-go image pinned
        services[nodename] = {
            "image": "ethereum/client-go:v1.13.14",
            "volumes": [f"./{datadir}:/root/.ethereum"],
            "ports": [f"{9636+i-1}:9636", f"{30303+i-1}:30303"],
            "command": f"--http --http.addr 0.0.0.0 --http.port 9636 --http.api eth,net,web3,personal --networkid {chain_id} --datadir /root/.ethereum"
        }

    compose = {"version":"3.8","services":services}
    compose_path = os.path.join(outdir, "docker-compose.yml")
    with open(compose_path, "w") as f:
        yaml_text = "version: '3.8'\nservices:\n"
        for name, cfg in services.items():
            f.write(f"# Service: {name}\n")
        # write simple docker-compose via python to keep dependency-free
    # For robustness, also write a helper script to initialize nodes using geth init
    init_sh = os.path.join(outdir, "init_nodes.sh")
    with open(init_sh, "w") as f:
        f.write("#!/bin/sh\nset -e\n")
        f.write("echo 'Initialize nodes with genesis'\n")
        for i in range(1, num_nodes+1):
            nodedir = f"./data/gbtnode{i}"
            f.write(f"mkdir -p {nodedir}\n")
            # copy genesis into datadir and run geth init via docker run (assuming image present)
            f.write(f"docker run --rm -v {os.path.abspath(outdir)}:/work -v {os.path.abspath(os.path.join(outdir, 'data', 'gbtnode'+str(i)))}:/root/.ethereum ethereum/client-go:v1.13.14 init /work/genesis.json\n")
    os.chmod(init_sh, 0o755)

    return outdir, genesis_path

def provision_cluster_worker(num_nodes:int=3, chain_id:int=CHAIN_ID):
    """Generates compose + genesis, saves genesis to memory server, optionally attempts to run docker compose"""
    outdir, genesis_path = generate_geth_node_compose(num_nodes, chain_id)
    # Save genesis to MEMORY_URL if available
    if MEMORY_URL:
        ok, resp = save_to_memory(f"genesis_{int(time.time())}.json", genesis_path)
        print("[provision] saved genesis to memory:", ok, resp)
    # Optionally try to run docker-compose (if DOCKER_RUN true and docker is available)
    if DOCKER_RUN:
        try:
            print("[provision] attempting to run docker compose in", outdir)
            # run docker compose up -d (best-effort)
            import subprocess
            subprocess.run(["docker", "compose", "up", "-d"], cwd=outdir, check=False)
        except Exception as e:
            print("[provision] docker compose run error:", e)
    return outdir

# ---------- REST endpoints ----------
@app.get("/", response_class=HTMLResponse)
def ui():
    with open("editor_index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/create_bot")
def create_bot(name: str = Form(...), persona: str = Form("You are a helpful assistant."), max_length: int = Form(128)):
    cfg = {"name": name, "persona": persona, "max_length": int(max_length)}
    # save locally
    cfg_path = os.path.join(BOTS_DIR, f"{name}.bot.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    # save to memory
    saved = False; resp_text = None
    if MEMORY_URL:
        ok, resp_text = save_to_memory(f"{name}.bot.json", cfg_path)
        saved = ok
    return {"created": name, "saved_to_memory": saved, "memory_resp": resp_text}

@app.post("/deploy_bot")
def deploy_bot(name: str = Form(...)):
    cfg_path = os.path.join(BOTS_DIR, f"{name}.bot.json")
    if not os.path.exists(cfg_path):
        # try load from memory
        if MEMORY_URL:
            r = requests.get(f"{MEMORY_URL}/ml/load/{name}.bot.json")
            if r.status_code == 200:
                cfg = json.loads(r.content.decode())
                with open(cfg_path,"w") as f: json.dump(cfg,f)
            else:
                raise HTTPException(status_code=404, detail="bot config not found locally or in memory")
        else:
            raise HTTPException(status_code=404, detail="bot config not found")
    with open(cfg_path) as f:
        cfg = json.load(f)
    botcfg = BotConfig(**cfg)
    inst = BotInstance(botcfg)
    with bots_lock:
        bots[name] = inst
    return {"deployed": name}

@app.post("/bot/{name}/chat")
def bot_chat(name: str, message: str = Form(...)):
    with bots_lock:
        inst = bots.get(name)
        if not inst:
            raise HTTPException(status_code=404, detail="bot not deployed")
    reply = inst.chat(message)
    # push history to memory
    hist_path = os.path.join(BOTS_DIR, f"{name}.history.json")
    with open(hist_path, "w") as f: json.dump(inst.get_history(), f)
    if MEMORY_URL:
        save_to_memory(f"{name}.history.json", hist_path)
    return {"reply": reply}

@app.post("/bot/{name}/chat_with_image")
async def bot_chat_image(name: str, message: str = Form(...), image: UploadFile = File(...)):
    # store image locally and to memory, then include placeholder text for bot
    img_path = os.path.join(UPLOADS_DIR, f"{int(time.time())}_{image.filename}")
    with open(img_path, "wb") as out:
        shutil.copyfileobj(image.file, out)
    if MEMORY_URL:
        _ok, _resp = save_to_memory(os.path.basename(img_path), img_path)
    # craft prompt that includes notice about image
    prompt = f"[Image uploaded: {os.path.basename(img_path)}]\n{message}"
    with bots_lock:
        inst = bots.get(name)
        if not inst:
            raise HTTPException(status_code=404, detail="bot not deployed")
    reply = inst.chat(prompt)
    # save history
    hist_path = os.path.join(BOTS_DIR, f"{name}.history.json")
    with open(hist_path, "w") as f: json.dump(inst.get_history(), f)
    if MEMORY_URL:
        save_to_memory(f"{name}.history.json", hist_path)
    return {"reply": reply, "image_saved": os.path.basename(img_path)}

@app.post("/generate_genesis")
def api_generate_genesis(chainId: int = CHAIN_ID, signer: str = Form(None), clique: bool = Form(True), gasLimit: int = Form(8000000)):
    try:
        gen = build_clique_genesis(chainId, signer, gasLimit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    # save locally and to memory
    path = os.path.join(OUT_DIR, f"genesis_{int(time.time())}.json")
    with open(path, "w") as f: json.dump(gen, f, indent=2)
    saved = False; resp_text = None
    if MEMORY_URL:
        ok, resp_text = save_to_memory(os.path.basename(path), path)
        saved = ok
    return {"genesis": gen, "saved_to_memory": saved, "memory_resp": resp_text, "path": path}

@app.post("/deploy_token")
def api_deploy_token(name: str = Form(...), symbol: str = Form(...), decimals: int = Form(18), total_supply: int = Form(...)):
    if PRIVATE_KEY is None:
        raise HTTPException(status_code=400, detail="PRIVATE_KEY not configured")
    if not w3.is_connected():
        raise HTTPException(status_code=500, detail="RPC unreachable")
    try:
        abi, bytecode = compile_source(ERC20_SRC, output_values=["abi","bin"]).popitem()[1].values()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"compile error: {e}")
    acct = w3.eth.account.from_key(PRIVATE_KEY)
    nonce = w3.eth.get_transaction_count(acct.address)
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx = contract.constructor(name, symbol, int(decimals), int(total_supply)).build_transaction({
        "from": acct.address,
        "nonce": nonce,
        "gas": 5_000_000,
        "gasPrice": w3.eth.gas_price,
        "chainId": CHAIN_ID
    })
    try:
        receipt = sign_and_send(tx, PRIVATE_KEY, wait_receipt=True, timeout=180)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"deploy failed: {e}")
    contract_address = receipt.contractAddress if hasattr(receipt, "contractAddress") else receipt.get("contractAddress")
    # save metadata to memory
    meta = {"name": name, "symbol": symbol, "decimals": decimals, "total_supply": total_supply, "address": contract_address, "tx": dict(receipt)}
    meta_path = os.path.join(OUT_DIR, f"token_{name}_{int(time.time())}.json")
    with open(meta_path, "w") as f: json.dump(meta, f, indent=2)
    saved = False; resp_text = None
    if MEMORY_URL:
        ok, resp_text = save_to_memory(os.path.basename(meta_path), meta_path)
        saved = ok
    return {"contract_address": contract_address, "receipt": dict(receipt), "saved_to_memory": saved, "memory_resp": resp_text}

@app.post("/provision_cluster")
def api_provision_cluster(num_nodes: int = Form(3), chain_id: int = Form(CHAIN_ID)):
    # spawn thread to provision
    def job():
        path = provision_cluster_worker(num_nodes, chain_id)
        print("[provision job] wrote to:", path)
    t = threading.Thread(target=job, daemon=True)
    t.start()
    return {"status":"started", "num_nodes": num_nodes}

@app.get("/bots")
def list_bots():
    with bots_lock:
        return {"bots": list(bots.keys())}

@app.get("/bot/{name}/history")
def bot_history(name: str):
    with bots_lock:
        inst = bots.get(name)
        if not inst:
            raise HTTPException(status_code=404, detail="not deployed")
        return {"history": inst.get_history()}

@app.get("/health")
def health():
    return {"status":"ok","rpc_connected": w3.is_connected(), "memory": bool(MEMORY_URL)}

# ---------- end of main.py ----------
