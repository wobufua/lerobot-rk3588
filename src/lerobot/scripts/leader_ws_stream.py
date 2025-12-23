# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
简化版 lerobot_record.py

功能：
- 连接主臂 (teleop = so100_leader / so101_leader / koch_leader 等)
- 通过 WebSocket 实时广播主臂关节数据给 ROS2（或其他客户端）
- 不再使用从臂、不再录制数据集、不再上传 Hub

启动示例：

python src/lerobot/scripts/lerobot_record.py \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=master \
  --fps=30
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import websockets

from lerobot.configs import parser
from lerobot.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
    koch_leader,
    homunculus,
    bi_so100_leader,
)
from lerobot.utils.utils import init_logging

# =========================
# WebSocket 服务端
# =========================

WS_CLIENTS = set()


async def ws_handler(websocket):
    print("✅ ROS client connected")
    WS_CLIENTS.add(websocket)
    try:
        async for _ in websocket:
            # 不需要处理客户端发来的消息，丢弃即可
            pass
    except Exception as e:
        print("❌ ROS client disconnected:", e)
    finally:
        WS_CLIENTS.remove(websocket)


async def ws_send_joint(data: dict[str, Any]):
    """把主臂关节字典广播给所有已连接的 WebSocket 客户端"""
    if not WS_CLIENTS:
        return
    msg = json.dumps(data)
    # 并发发送给所有客户端
    await asyncio.gather(*[ws.send(msg) for ws in WS_CLIENTS])


# =========================
# 配置：只保留 teleop + fps
# =========================

@dataclass
class StreamConfig:
    # 主臂配置（原来的 TeleoperatorConfig）
    teleop: TeleoperatorConfig
    # 推送频率（Hz）
    fps: int = 30
    # 预留字段，避免 parser 报错
    play_sounds: bool = False
    display_data: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# =========================
# 主逻辑：连接主臂并循环发送
# =========================

async def stream_loop(cfg: StreamConfig):
    """单个 asyncio 事件循环：同时跑 WebSocket server + 读取主臂动作并广播"""

    # 启动 WebSocket server
    server = await websockets.serve(ws_handler, "0.0.0.0", 8765)
    print("✅ WebSocket server running on ws://0.0.0.0:8765")

    # 创建并连接主臂
    teleop: Teleoperator = make_teleoperator_from_config(cfg.teleop)
    teleop.connect()
    print("✅ Teleop connected:", cfg.teleop.type, cfg.teleop.port)

    dt = 1.0 / max(cfg.fps, 1)

    try:
        while True:
            start_t = time.perf_counter()

            # 从主臂读取当前动作（关节位置字典）
            act = teleop.get_action()
            print(act)
            # act 例如：{"shoulder_pan.pos": ..., "shoulder_lift.pos": ..., "gripper.pos": ...}

            # 通过 WebSocket 广播
            await ws_send_joint(act)

            # 控制频率
            elapsed = time.perf_counter() - start_t
            await asyncio.sleep(max(dt - elapsed, 0.0))

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    finally:
        print("🔌 Closing teleop and websocket server")
        teleop.disconnect()
        server.close()
        await server.wait_closed()


@parser.wrap()
def main(cfg: StreamConfig):
    """入口函数：只负责解析配置，然后跑 asyncio 循环"""
    init_logging()
    logging.info("StreamConfig: %s", cfg)
    asyncio.run(stream_loop(cfg))


if __name__ == "__main__":
    main()
