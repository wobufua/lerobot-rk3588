import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import Float64MultiArray

import asyncio
import websockets
import json
import threading
import cv2
import numpy as np


WS_SERVER = "ws://172.20.10.3:8765"


class WebSocketJointNode(Node):
    def __init__(self):
        super().__init__('websocket_joint_and_camera_node')

        self.joint_state_pub = self.create_publisher(
            JointState, '/joint_states', 10)

        self.raw_pub = self.create_publisher(
            Float64MultiArray, '/so100/raw_angles', 10)

        # ⭐ 改成 CompressedImage
        self.create_subscription(
            CompressedImage,
            "/master/mid_camera",
            self.cb_mid,
            10
        )

        self.get_logger().info("✅ WebSocket + Master Mid Camera Node started")

    def cb_mid(self, msg: CompressedImage):
        img_array = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("❌ Failed to decode compressed image")
            return

        cv2.imshow("Master Mid Camera", frame)
        cv2.waitKey(1)

    def publish_joint_state(self, data: dict):
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = list(data.keys())
        joint_msg.position = [float(v) for v in data.values()]

        self.joint_state_pub.publish(joint_msg)

        raw_msg = Float64MultiArray()
        raw_msg.data = [float(v) for v in data.values()]
        self.raw_pub.publish(raw_msg)


# WebSocket 部分保持不变
async def websocket_client(node: WebSocketJointNode):
    while rclpy.ok():
        try:
            async with websockets.connect(WS_SERVER) as ws:
                while rclpy.ok():
                    msg = await ws.recv()
                    data = json.loads(msg)
                    node.publish_joint_state(data)
        except Exception:
            await asyncio.sleep(3)


def run_websocket(node):
    asyncio.run(websocket_client(node))


def main(args=None):
    rclpy.init(args=args)
    node = WebSocketJointNode()

    threading.Thread(
        target=run_websocket,
        args=(node,),
        daemon=True
    ).start()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
