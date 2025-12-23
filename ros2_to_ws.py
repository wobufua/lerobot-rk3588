import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import Header

import socket
import json
import threading
import cv2
import time


# =============================
# TCP 配置
# =============================
TCP_BIND_HOST = "172.20.10.2"   # ⭐ 推荐监听所有网卡
TCP_ANGLE_PORT = 9002


class MasterROSBridge(Node):
    def __init__(self):
        super().__init__('master_ros_bridge')

        # =============================
        # ROS2 joint_states 订阅
        # =============================
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_callback,
            10
        )

        # =============================
        # TCP server（独立线程）
        # =============================
        self.angle_client = None
        self.tcp_thread = threading.Thread(
            target=self.tcp_server_loop,
            daemon=True
        )
        self.tcp_thread.start()

        # =============================
        # mid 相机 ROS2（压缩）发布
        # =============================
        self.cam_pub = self.create_publisher(
            CompressedImage,
            "/master/mid_camera",
            10
        )

        self.mid_cam_index = 22
        self.cap = cv2.VideoCapture(self.mid_cam_index)

        if not self.cap.isOpened():
            self.get_logger().error(
                f"❌ Cannot open mid camera index={self.mid_cam_index}"
            )
        else:
            self.get_logger().info(
                f"🎥 mid camera opened at index {self.mid_cam_index}"
            )

        # ⭐ ROS2 Timer（~30 FPS）
        self.create_timer(0.03, self.publish_mid_image)

        self.get_logger().info("✅ Master ROS2 Bridge (Compressed Image) started")

    # ============================================================
    # TCP server loop（独立线程）
    # ============================================================
    def tcp_server_loop(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((TCP_BIND_HOST, TCP_ANGLE_PORT))
        server.listen(1)

        self.get_logger().info(
            f"🔌 TCP Angle Server listening on {TCP_BIND_HOST}:{TCP_ANGLE_PORT}"
        )

        while rclpy.ok():
            try:
                client, addr = server.accept()
                self.angle_client = client
                self.get_logger().info(f"✅ TCP client connected: {addr}")
            except Exception:
                time.sleep(0.1)

    # ============================================================
    # joint_states → TCP
    # ============================================================
    def joint_callback(self, msg: JointState):
        now = self.get_clock().now().nanoseconds / 1e9
        angles = {n: float(v) for n, v in zip(msg.name, msg.position)}

        self.get_logger().info(
            f"[JOINT RX @ {now:.3f}s] {angles}"
        )

        if self.angle_client is None:
            return

        try:
            self.angle_client.sendall(
                (json.dumps(angles) + "\n").encode("utf-8")
            )
        except Exception:
            self.get_logger().warn("❌ TCP client disconnected")
            self.angle_client = None

    # ============================================================
    # mid camera → ROS2 CompressedImage
    # ============================================================
    def publish_mid_image(self):
        if not self.cap.isOpened():
            return

        ok, frame = self.cap.read()
        if not ok:
            return

        # ⭐ JPEG 压缩（质量 70，推荐跨网）
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 10]
        )
        if not success:
            self.get_logger().warn("❌ Image compression failed")
            return

        msg = CompressedImage()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "master_mid_cam"
        msg.format = "jpeg"
        msg.data = encoded.tobytes()

        self.cam_pub.publish(msg)


# ============================================================
# main
# ============================================================
def main(args=None):
    print("\n🚀 Master ROS2 Bridge (JOINT + MID CAMERA / COMPRESSED)\n")
    rclpy.init(args=args)

    node = MasterROSBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()