#!/usr/bin/env python3
"""
ros_publisher_node.py

Centralized ROS2 publisher node used by all triggers.
Implements singleton access (thread-safe) so that only one node ever exists.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading


class ROS2PublisherNode(Node):
    _instance = None
    _lock = threading.Lock()

    def __init__(self, name="realtime_capture_node"):
        super().__init__(name)
        self.untouched_piece_data = ""
        self.handspeed_piece_data = ""
        self.emotion_piece_data = ""

        # Publishers
        self.untouched_pub = self.create_publisher(String, '/untouched_piece', 10)
        self.handspeed_pub = self.create_publisher(String, '/hand_speed', 10)
        self.emotion_pub = self.create_publisher(String, '/emotion', 10)

        # Timer for publishing at fixed rate (1Hz)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        # Only publish if data is available
        if self.untouched_piece_data:
            msg = String()
            msg.data = self.untouched_piece_data
            self.untouched_pub.publish(msg)

        if self.handspeed_piece_data:
            msg = String()
            msg.data = self.handspeed_piece_data
            self.handspeed_pub.publish(msg)

        if self.emotion_piece_data:
            msg = String()
            msg.data = self.emotion_piece_data
            self.emotion_pub.publish(msg)

    @classmethod
    def get_instance(cls):
        """Get or create the singleton ROS2 node safely."""
        with cls._lock:
            if cls._instance is None:
                if not rclpy.ok():
                    rclpy.init()
                cls._instance = cls()
        return cls._instance

    @classmethod
    def shutdown(cls):
        """Safely shut down the singleton node."""
        if cls._instance is not None:
            try:
                cls._instance.destroy_node()
            except Exception:
                pass
            cls._instance = None
        if rclpy.ok():
            rclpy.shutdown()


def main():
    rclpy.init()
    node = ROS2PublisherNode.get_instance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally :
        ROS2PublisherNode.shutdown()


if __name__ == "__main__":
    main()