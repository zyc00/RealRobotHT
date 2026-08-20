#!/usr/bin/env bash
# Keep the Quest awake with the headset off (proximity-sensor sleep disabled).
#
#   quest_awake.sh          # stay awake  (verified working on Quest 3)
#   quest_awake.sh off      # restore normal proximity behaviour
#
# NOT persistent across a headset reboot - re-run after each boot.
set -euo pipefail
export PATH="$HOME/Android/Sdk/platform-tools:$PATH"

if [[ "${1:-on}" == "off" ]]; then
    adb shell am broadcast -a com.oculus.vrpowermanager.prox_open >/dev/null
    echo "proximity sensor re-enabled (headset sleeps when taken off)"
else
    adb shell am broadcast -a com.oculus.vrpowermanager.prox_close >/dev/null
    echo "proximity sleep disabled - headset stays awake off-head"
fi
sleep 1
adb shell dumpsys power 2>/dev/null | grep -oE "mWakefulness=[A-Za-z]+" | head -1
