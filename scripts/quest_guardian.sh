#!/usr/bin/env bash
# Pause the Quest Guardian/boundary system.
#
# Moving a placed headset makes Quest think you left the play area, so it drops
# to passthrough and pauses the app.  This suppresses that.
#
#   quest_guardian.sh          # pause guardian
#   quest_guardian.sh off      # restore it
#
# NOT persistent across a headset reboot.  If it does not take effect, turn
# Boundary off in the headset: Settings > Physical Space (or Guardian) >
# Boundary Off, which requires Developer Mode (already enabled here).
set -euo pipefail
export PATH="$HOME/Android/Sdk/platform-tools:$PATH"

if [[ "${1:-on}" == "off" ]]; then
    adb shell setprop debug.oculus.guardian_pause 0 || true
    adb shell setprop debug.oculus.guardian_disable 0 || true
    echo "guardian restored"
else
    adb shell setprop debug.oculus.guardian_pause 1 || true
    adb shell setprop debug.oculus.guardian_disable 1 || true
    echo "guardian paused"
fi
echo -n "  guardian_pause="; adb shell getprop debug.oculus.guardian_pause
echo -n "  guardian_disable="; adb shell getprop debug.oculus.guardian_disable
