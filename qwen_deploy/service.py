"""Qwen2.5-3B-Instruct API Service for Drone Task Decomposition on H100.

Provides:
1. /api/decompose: Direct natural language instruction -> Structured Aerial-WAM JSON.
2. /v1/chat/completions: OpenAI-compatible chat endpoint.
3. /health: Service health check.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen_service")

app = FastAPI(title="Qwen2.5-3B Task Decomposition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = None
model = None

DECOMPOSE_SYSTEM_PROMPT = """你是一个专业的无人机战术任务规划助理。你的职责是将用户的自然语言任务指令精准解析为严格的 JSON 结构化空间参数与视觉提示词。

请务必直接输出合法的 JSON 字符串，不要包含任何额外的问候、解释或 markdown 以外的闲聊文字。

输出 JSON Schema 规范：
{
  "task_type": "search_and_follow" | "search_only" | "patrol",
  "search_area": {
    "center_xy": [x, y],          // 相对当前位置或世界坐标中心 [float, float]
    "radius_m": float,            // 搜寻半径或半边长（米，默认 30.0）
    "altitude_m": float           // 巡航高度（米，默认 25.0）
  },
  "target": {
    "visual_prompt": string,      // 供开放词表检测器（如 YOLO-World/Grounding DINO）使用的精准描述
    "category": string            // 基础类别英文名: car / truck / person / bicycle / bus 等
  },
  "follow_config": {
    "standoff_dist_m": float,     // 伴飞后方距离（米，默认 6.0）
    "standoff_height_m": float    // 伴飞相对高度（米，默认 3.0）
  }
}"""


class DecomposeRequest(BaseModel):
    instruction: str = Field(..., description="User's natural language drone flight instruction")
    drone_current_pos: Optional[List[float]] = Field(default=[0.0, 0.0, 20.0], description="Current drone position [x, y, z]")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "qwen2.5:3b"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512


@app.on_event("startup")
def load_model():
    global tokenizer, model
    logger.info(f"Loading {MODEL_NAME} onto CUDA on H100...")
    try:
        from modelscope import snapshot_download
        model_dir = snapshot_download(MODEL_NAME)
    except Exception as e:
        logger.warning(f"ModelScope snapshot download failed ({e}), falling back to HF model id")
        model_dir = MODEL_NAME

    logger.info(f"Model path: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Qwen2.5-3B-Instruct successfully loaded on H100 GPU!")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "gpu_memory_allocated_gb": f"{torch.cuda.memory_allocated(0)/1024**3:.2f} GB" if torch.cuda.is_available() else "0",
    }


def generate_text(messages: List[Dict[str, str]], temperature: float = 0.2, max_new_tokens: int = 512) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.01,
            top_p=0.9,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


@app.post("/api/decompose")
def decompose_instruction(req: DecomposeRequest):
    """Decompose natural language instruction into Aerial-WAM structured flight JSON."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    t0 = time.time()
    user_content = f"无人机当前坐标: {req.drone_current_pos}\n用户任务指令: {req.instruction}"
    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    raw_output = generate_text(messages, temperature=0.1, max_new_tokens=384)
    elapsed = time.time() - t0

    # Extract JSON from potential code fences
    cleaned = raw_output
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    try:
        parsed_json = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback regex search
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if match:
            try:
                parsed_json = json.loads(match.group(0))
            except Exception:
                parsed_json = {"raw_output": raw_output, "error": "JSON parse error"}
        else:
            parsed_json = {"raw_output": raw_output, "error": "No JSON block found"}

    return {
        "status": "success",
        "instruction": req.instruction,
        "elapsed_ms": round(elapsed * 1000, 2),
        "plan": parsed_json,
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """OpenAI compatible chat completions endpoint."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    t0 = time.time()
    reply = generate_text(messages, temperature=req.temperature or 0.2, max_new_tokens=req.max_tokens or 512)
    elapsed = time.time() - t0

    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or "qwen2.5:3b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "elapsed_ms": round(elapsed * 1000, 2),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
