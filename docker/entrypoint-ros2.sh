#!/bin/bash
# ワークスペースは起動時にビルドする (bind mount された ros2_ws/ を使うため。
# docker/entrypoint-gpu.sh の「起動時に作る」と同じ考え方)。
set -e
source /opt/ros/humble/setup.bash
if [ -d /workspace/ros2_ws/src ]; then
    cd /workspace/ros2_ws
    colcon build --symlink-install
    source /workspace/ros2_ws/install/setup.bash
fi
cd /workspace
exec "$@"
