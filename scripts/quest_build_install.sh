#!/usr/bin/env bash
# Build and install the RoboVR Quest client.
#
# Toolchain set up 2026-08-19 without sudo:
#   JDK 17      conda env "androidbuild"      (conda create -n androidbuild -c conda-forge openjdk=17)
#   Android SDK ~/Android/Sdk                 (cmdline-tools + platform 35, build-tools 35.0.0,
#                                              ndk 26.1.10909125, cmake 3.22.1)
#   OpenXR      RoboVR submodule at release-1.1.59 (git submodule update --init --recursive)
set -euo pipefail

export JAVA_HOME="${JAVA_HOME:-$HOME/miniforge3/envs/androidbuild}"
export ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
export PATH="$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$PATH"

ROBOVR="${ROBOVR:-$HOME/projects/RoboVR}"
PORT="${PORT:-7777}"
APK="$ROBOVR/quest_client/app/build/outputs/apk/debug/app-debug.apk"

echo "sdk.dir=$ANDROID_HOME" > "$ROBOVR/quest_client/local.properties"

if [[ "${1:-}" != "--install-only" ]]; then
  echo "== building =="
  (cd "$ROBOVR/quest_client" && ./gradlew assembleDebug --no-daemon)
fi

echo "== devices =="
adb devices -l
if ! adb devices | grep -qE '\sdevice$'; then
  echo
  echo "No AUTHORISED device. Put the headset on and accept 'Allow USB debugging'." >&2
  echo "If it shows 'unauthorized', that prompt is waiting inside the headset." >&2
  exit 1
fi

echo "== installing =="
adb install -r "$APK"
adb reverse "tcp:${PORT}" "tcp:${PORT}"
adb shell am start -n com.yuchen.robovr/.VrActivity
echo "launched; adb reverse tcp:${PORT} is up"
