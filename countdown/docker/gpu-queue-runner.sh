#!/usr/bin/env bash
# One GPU job at a time, lowest-numbered pending script first.
# Add work by dropping NN-name.sh into pending/.
#
# Host (fallback): setsid nohup bash docker/gpu-queue-runner.sh &  -- stops at pending/STOP.
# Container:      docker compose -f docker/compose.yaml up -d gpu-queue
#                 QUEUE_DAEMON=1: never stops on its own; idles until work arrives.
#
# Crash safety: a job sits in running/ while it runs.  If the runner dies mid-job (reboot,
# container restart, session teardown), the next start moves it back to pending/, and the
# job's own caches (per-fold predictions in runs/ensemble/) make the re-run resume.
cd /home/counterbitio/CountDown/countdown
Q=runs/gpu-queue
mkdir -p $Q/pending $Q/running $Q/done $Q/logs

for job in $Q/running/*.sh; do
  [ -e "$job" ] || continue
  echo "$(date '+%F %T') REQUEUE $(basename "$job" .sh) (interrupted)" >> $Q/queue.log
  mv "$job" $Q/pending/
done

# The PCIe/power logger needs host sudo; inside the container it is run on the host instead.
if [ -z "$IN_CONTAINER" ] && sudo -n true 2>/dev/null; then
  sudo -n /home/counterbitio/gpu-watch/watch.sh > $Q/logs/gpu-watch-$(date +%Y%m%d-%H%M%S).out 2>&1 &
  WPID=$!
  trap 'sudo -n kill -TERM $WPID 2>/dev/null' EXIT
fi

xids() {
  # Xid lines from the NVIDIA driver ("NVRM: Xid (PCI:..."), not the r8169 NIC's "XID".
  if [ -n "$IN_CONTAINER" ]; then echo "n/a in container"; else
    journalctl -k -b 0 --no-pager | grep -ciE 'nvrm: xid'; fi
}

echo "$(date '+%F %T') RUNNER START ${IN_CONTAINER:+(container)}" >> $Q/queue.log
while :; do
  job=$(ls $Q/pending/*.sh 2>/dev/null | sort | head -1)
  if [ -z "$job" ]; then
    if [ -z "$QUEUE_DAEMON" ] && [ -e $Q/pending/STOP ]; then rm -f $Q/pending/STOP; break; fi
    sleep 20; continue
  fi
  name=$(basename "$job" .sh)
  echo "$(date '+%F %T') START $name" >> $Q/queue.log
  mv "$job" $Q/running/
  bash $Q/running/$name.sh > $Q/logs/$name.log 2>&1
  rc=$?   # before any $(...): a command substitution resets $?
  mv $Q/running/$name.sh $Q/done/
  echo "$(date '+%F %T') EXIT $rc $name (Xid this boot: $(xids))" >> $Q/queue.log
done
echo "$(date '+%F %T') RUNNER STOPPED" >> $Q/queue.log
