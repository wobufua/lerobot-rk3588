import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class RecordControlNode(Node):
    def __init__(self, events):
        super().__init__('lerobot_record_control')
        self.events = events

        self.sub = self.create_subscription(
            String,
            '/lerobot/record_cmd',
            self.on_cmd,
            10
        )

        self.pub = self.create_publisher(
            String,
            '/lerobot/record_status',
            10
        )

    def on_cmd(self, msg):
        try:
            data = json.loads(msg.data)
            cmd = data.get("cmd")
        except Exception:
            self.get_logger().warn("Invalid cmd")
            return

        if cmd == "STOP_EPISODE":
            self.events["exit_early"] = True
            self.get_logger().info("STOP_EPISODE")

        elif cmd == "STOP_ALL":
            self.events["stop_recording"] = True
            self.get_logger().info("STOP_ALL")

        elif cmd == "SAVE_AND_EXIT":
            self.events["stop_recording"] = True
            self.get_logger().info("SAVE_AND_EXIT")

    def publish_status(self, status_dict):
        msg = String()
        msg.data = json.dumps(status_dict)
        self.pub.publish(msg)