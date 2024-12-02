#!/usr/bin/env bash
set -eu

# REF: https://index.ros.org/doc/ros2/Installation/Linux-Install-Debians/
# by Open Robotics, licensed under CC-BY-4.0
# source: https://github.com/ros2/ros2_documentation

# REF: https://github.com/Tiryoh/ros2_setup_scripts_ubuntu/blob/master/run.sh
# by https://github.com/Tiryoh, Apache-2.0 License

# REF https://gbiggs.github.io/rosjp_ros2_intro/computer_prep_linux.html

CHOOSE_ROS_DISTRO=foxy # or dashing
INSTALL_PACKAGE=desktop # or ros-base

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo -E apt-get install -yq \
		curl \
		gnupg2 \
		lsb-release \
		bash-completion \
		build-essential \
		locales \
		tmux \
		x11-apps \
		eog && \
		sudo rm -rf /var/lib/apt/lists/*

sudo apt-get install -yq \
        google-mock \
        libceres-dev \
        liblua5.3-dev \
        libboost-dev \
        libboost-iostreams-dev \
        libprotobuf-dev \
        protobuf-compiler \
        libcairo2-dev \
        libpcl-dev \
        python-sphinx && \
		sudo rm -rf /var/lib/apt/lists/*

# Install ROS2 packages
curl -Ls https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo apt-key add -
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
sudo apt-get  update
sudo apt-get  install -y ros-$CHOOSE_ROS_DISTRO-$INSTALL_PACKAGE
sudo apt-get  install -y python3-argcomplete
sudo apt-get  install -y python3-colcon-common-extensions
if [ "$(lsb_release -cs)" = "bionic" ]; then
	sudo apt-get  install -y python-rosdep python3-vcstool # https://index.ros.org/doc/ros2/Installation/Linux-Development-Setup/
elif [ "$(lsb_release -cs)" = "focal" ]; then
	sudo apt-get  install -y python3-rosdep python3-vcstool # https://index.ros.org/doc/ros2/Installation/Linux-Development-Setup/
fi

sudo apt-get install -yq \
		ros-$CHOOSE_ROS_DISTRO-gazebo-ros-* \
		ros-$CHOOSE_ROS_DISTRO-turtlesim \
        ros-$CHOOSE_ROS_DISTRO-cartographer \
        ros-$CHOOSE_ROS_DISTRO-cartographer-ros \
        ros-$CHOOSE_ROS_DISTRO-navigation2 \
        ros-$CHOOSE_ROS_DISTRO-nav2-bringup

# Add ROS2 env values in bash
grep -F "source /opt/ros/$CHOOSE_ROS_DISTRO/setup.bash" ~/.bashrc ||
echo "source /opt/ros/$CHOOSE_ROS_DISTRO/setup.bash" >> ~/.bashrc

set +u

# Set ROS2 env values
source /opt/ros/$CHOOSE_ROS_DISTRO/setup.bash

echo "success installing ROS2 $CHOOSE_ROS_DISTRO"
echo "Run 'source /opt/ros/$CHOOSE_ROS_DISTRO/setup.bash'"

