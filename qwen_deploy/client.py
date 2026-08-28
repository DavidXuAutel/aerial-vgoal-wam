"""Test client for Qwen2.5-3B instruction decomposition on H100."""

import json
import requests
import sys

API_URL = "http://localhost:8000/api/decompose"
HEALTH_URL = "http://localhost:8000/health"

TEST_INSTRUCTIONS = [
    "去前方 50 米外的东侧树林，搜寻一辆蓝色皮卡车并保持 8 米距离跟着它",
    "在当前位置半径 40 米区域内盘旋巡航，寻找穿红色外套的行人",
    "飞往坐标 [30, -20]，搜寻一辆白色面包车并后方 6 米高度 3 米伴飞",
    "Search the industrial zone 100m north for a yellow delivery van and follow it closely",
]


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    print(f"Connecting to Qwen Service at {base_url}...")

    try:
        health = requests.get(f"{base_url}/health", timeout=5).json()
        print(f"Health status: {health}\n")
    except Exception as e:
        print(f"Failed to connect to health endpoint: {e}")
        return

    for idx, inst in enumerate(TEST_INSTRUCTIONS, 1):
        print(f"--- Test Case {idx} ---")
        print(f"Input: \"{inst}\"")
        payload = {
            "instruction": inst,
            "drone_current_pos": [0.0, 0.0, 20.0],
        }
        resp = requests.post(f"{base_url}/api/decompose", json=payload, timeout=10)
        if resp.status_code == 200:
            res_data = resp.json()
            print(f"Latency: {res_data['elapsed_ms']} ms")
            print("Decomposed Plan:")
            print(json.dumps(res_data["plan"], indent=2, ensure_ascii=False))
        else:
            print(f"Error {resp.status_code}: {resp.text}")
        print()


if __name__ == "__main__":
    main()
