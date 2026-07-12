# SSD Health, Wear Monitoring & Wear Reduction (Ubuntu / Linux)

> Scope: NVMe + SATA SSDs on Ubuntu. Commands assume `sudo` where device access
> is required. Most distros split `smartctl` into the `smartmontools` package and
> `nvme` into `nvme-cli`.

---

## 0. Install prerequisites

```bash
sudo apt update
sudo apt install -y smartmontools nvme-cli
```

`smartmontools` provides `smartctl`. `nvme-cli` provides the `nvme` command for
NVMe-specific admin passthrough (some attributes aren't surfaced by `smartctl`).

---

## 1. Identify your disks

Before checking anything, know what you have:

```bash
# List block devices with topology + transport (nvme vs sata/sata)
lsblk -d -o NAME,TRAN,ROTA,SIZE,MODEL

# Kernel-reported disk names
ls /dev/disk/by-id/

# Filesystem mounts (to see what is on which disk, and mount options)
findmnt -D
mount | grep -E 'ext4|btrfs|xfs|f2fs'
```

`ROTA=0` => solid state. `TRAN=nvme` => NVMe; `TRAN=sata` => SATA/AHCI.

---

## 2. Check SSD wear

### 2a. NVMe — wear odometer

```bash
# Full SMART health log
sudo smartctl -a /dev/nvme0

# NVMe-native view (often clearer)
sudo nvme smart-log /dev/nvme0
```

Key fields (NVMe):

| Field | Meaning |
|---|---|
| `Percentage Used` | Vendor's wear estimate, starts at 0, reaches 100 at end-of-life. **Can exceed 100** if you keep using it. |
| `Available Spare` | % of spare blocks left (e.g. `100%`). Drops as cells wear. |
| `Available Spare Threshold` | When `Available Spare` falls below this, the drive flags a failure. |
| `Critical Warning` | Bitfield; `0x01` = available spare below threshold. |
| `Power On Hours` | Lifetime powered-on time. |
| `Data Units Written` | Host writes in 512KB units. Multiply by 512000 for bytes-ish. |
| `Media and Data Integrity Errors` | Non-zero is a real concern. |

One-liner to extract just the odometer:

```bash
sudo smartctl -a /dev/nvme0 | grep -i 'percentage used'
sudo nvme smart-log /dev/nvme0 | grep -i 'percentage_used'
```

### 2b. SATA — wear-leveling & total writes

```bash
# Full SMART report
sudo smartctl -a /dev/sda

# Human summary line
sudo smartctl -H /dev/sda      # "SMART overall-health: PASSED"
```

Key attributes for SATA SSDs (raw values matter, not the normalized 100→0 scale):

| Attribute | What it tells you |
|---|---|
| `Wear_Leveling_Count` (ID 177/173, Samsung/SK Hynix) | Normalized 100→0 = life left. Raw = average erase count per cell. |
| `Total_LBAs_Written` (ID 241/169) | Cumulative host writes. Multiply raw by 512 bytes for total bytes written. |
| `Power_On_Hours` (ID 9) | Lifetime hours. |
| `Reallocated_Sector_Ct` (ID 5) | Bad cells remapped. Non-zero and rising = degradation. |
| `Reallocated_Event_Count` (ID 196) | Remap events. |
| `Runtime_Bad_Block` (ID 183) | Factory + grown bad blocks. |
| `Percentage Used` (ID 202/231, Intel/SandForce) | Some SATA SSDs expose an NVMe-style % used. |

Extract just the wear signals:

```bash
sudo smartctl -a /dev/sda | grep -Ei 'wear_leveling|total_lbas_written|reallocated|power_on_hours|percentage used|available spare'
```

Convert `Total_LBAs_Written` to human bytes:

```bash
raw=$(sudo smartctl -a /dev/sda | awk -F'[' '/Total_LBAs_Written/{gsub(/[ \t]+/,"",$10); print $10}')
# raw value is the last numeric column; multiply by 512
# simpler: use -x (json-ish) or just compute in awk
sudo smartctl -a /dev/sda | awk '/Total_LBAs_Written/{print "Bytes written ~", $10*512}'
```

> Note: the "disk grand" / "dismax" columns from the talk are a transcription of
> **`Discard_Granularity`** and **`Discard_Max`** — see §3.

### 2c. Continuous SMART monitoring (background)

```bash
# Enable SMART if off
sudo smartctl -s on /dev/sda

# Run the self-test (short, ~2 min)
sudo smartctl -t short /dev/sda
# Long test (hours): sudo smartctl -t long /dev/sda
# View results after it finishes:
sudo smartctl -l selftest /dev/sda

# Daemonized periodic scans (sends mail on threshold breach)
sudo systemctl enable --now smartd
sudoedit /etc/smartd.conf   # e.g. DEVICESCAN -a -m root@localhost -M daily
```

---

## 3. Verify TRIM / Discard support

TRIM lets the SSD reclaim erased blocks. Two things matter: the drive must
support `discard`, and the kernel must issue it.

### 3a. Does the drive advertise discard?

```bash
# Queue discard granularity + max, and whether discard is supported
cat /sys/block/sda/queue/discard_granularity   # bytes; 0 = NOT supported
cat /sys/block/sda/queue/discard_max_bytes      # largest single discard
cat /sys/block/sda/queue/discard_zeroes_data    # 1 = reads back as zeros
```

- `discard_granularity` **non-zero** (e.g. 512, 4096) => the device reports a
  discard alignment → discard is meaningful.
- `discard_granularity == 0` => the device does not support discard.

For NVMe the same files exist under `/sys/block/nvme0n1/queue/`.

### 3b. Is the mounted FS actually using discard?

```bash
# Mount options — look for 'discard' (continuous) or absence of it
findmnt -o TARGET,SOURCE,FSTYPE,OPTIONS | grep -E 'sda|nvme'

# Does the filesystem allow FITRIM ioctl?
sudo fstrim -v /      # if it frees bytes, discard works end-to-end
```

---

## 4. Scheduled trimming (recommended) — replace continuous discard

Use a **periodic** `fstrim`, NOT the `discard` mount option. Most modern Ubuntu
already ships this via `systemd` timer.

```bash
# Run a one-shot trim across all mounted filesystems (reports freed bytes)
sudo fstrim -a -v

# Ubuntu/Fedora/Debian ship a timer — enable + check it
sudo systemctl enable --now fstrim.timer
systemctl list-timers fstrim.timer     # next run, usually weekly (Mon 00:00)

# Force a run now
sudo systemctl start fstrim.service
journalctl -u fstrim.service --no-pager | tail -n 20
```

The default `fstrim.timer` fires **weekly** (Mon 00:00) — exactly the cadence the
talk recommends. Weekly is plenty for consumer drives; daily only helps if you
write tens of GB/day.

Cron alternative if you prefer it over the timer:

```bash
sudo crontab -e
# m h  dom mon dow   command
30 3 * * 0  /sbin/fstrim -a -v >> /var/log/fstrim.log 2>&1
```

---

## 5. Remove continuous discard from /etc/fstab (the warning)

Continuous `discard` issues a TRIM on every delete — extra writes/queueing and
unpredictable latency. Switch to scheduled `fstrim` (§4).

```bash
# Inspect current fstab
grep -E 'discard|ext4|f2fs|xfs|btrfs' /etc/fstab
```

If a line has `discard`, remove just that option:

```bash
# Before:
# UUID=xxxx  /  ext4  defaults,discard  0 1
# After (edit with your editor of choice):
# UUID=xxxx  /  ext4  defaults  0 1
sudoedit /etc/fstab
```

Then re-mount without reboot:

```bash
sudo mount -o remount /
# confirm discard is gone
findmnt -o TARGET,OPTIONS | grep ' /'
```

Do **not** put `discard` on `swap` or `tmpfs` lines either.

---

## 6. What decreases wear on a SATA SSD (Ubuntu)

Wear is driven by **write amplification**: every byte the host writes can be
amplified by the controller's garbage collection, over-provisioning churn, and
journaling. Reduce host writes and let the controller work efficiently.

### 6a. Reduce writes at the source

```bash
# 1. Move volatile/junk data to tmpfs (RAM) — no SSD writes at all.
#    Add to /etc/fstab:
#    tmpfs  /tmp        tmpfs  defaults,noatime,size=2G  0 0
#    tmpfs  /var/tmp    tmpfs  defaults,noatime,size=1G  0 0
sudoedit /etc/fstab

# 2. Quiet the logging noise (journald) — cap disk usage.
sudoedit /etc/systemd/journald.conf
#   SystemMaxUse=500M
#   SystemKeepFree=1G
#   MaxRetentionSec=2week
sudo systemctl restart systemd-journald
```

### 6b. Mount options that cut metadata writes

`noatime` (or `relatime`) stops the kernel from rewriting every file's access
time on every read — a big write saver for read-heavy workloads.

```bash
# In /etc/fstab change 'defaults' on your SSD mounts to:
#   defaults,noatime
# (Ubuntu already uses relatime by default, but noatime is stronger.)
sudoedit /etc/fstab
sudo mount -o remount,noatime /
```

### 6c. Let the SSD over-provision itself (free space = spare)

Free space is the controller's scratch pad for wear leveling. Keep headroom:

```bash
# Aim to keep >= 10–20% free on the SSD. For max endurance, leave 20%+.
df -h /

# Optional: create a static over-provision partition at the end of the disk
# (reclaim it as unpartitioned space the controller quietly uses as spare).
# Use gparted / parted to shrink the last partition, leaving ~10–20% unallocated.
sudo parted /dev/sda print
```

Some vendors ship a tool (Samsung `magician`, Kingston `SSD Manager`) to set
OP explicitly; on Linux, leaving unpartitioned space achieves the same effect.

### 6d. Filesystem choice & tuning

- **ext4** (default, safe): enable `discard` via timer only (§4), use `noatime`.
- **f2fs**: flash-native, good for pure-SSD small devices; less common on servers.
- **btrfs**: avoid on low-end SATA SSDs unless you need CoW/snapshots — CoW
  amplifies writes. If you use it, disable CoW on high-write dirs:
  ```bash
  sudo chattr +C /var/lib/postgresql /var/lib/docker   # nodatacow
  ```
### 6e. Stop redundant write workloads

```bash
# Browser/profile caches, package manager caches, container layers, build dirs
# are the usual silent writers. Point heavy ones at tmpfs or a spinning disk.
# e.g. move ~/.cache for heavy apps, or build in /tmp.

# Disable monthly fsck on ext4 (it's not a wear issue but avoids churn):
sudo tune2fs -c 0 -i 0 /dev/sda1    # never auto-fsck by mount count/time
```

### 6f. Keep firmware current

```bash
# Check current firmware
sudo smartctl -i /dev/sda | grep -i firmware
# Vendor tools (Linux support varies):
#   Samsung:    sudo nvme fw-log /dev/nvme0  (NVMe only)
#   Most SATA:  use the vendor Windows ISO / LiveUSB to flash.
# Visit the vendor site for the .iso; a USB Live boot is the reliable path.
```

Firmware updates fix garbage-collection bugs that can silently inflate wear.

### 6g. Hibernation writes the whole RAM image to disk

If you hibernate (`systemctl hibernate`), the kernel writes RAM→SSD every time.
Prefer **suspend-to-RAM** (`systemctl suspend`) which keeps power but writes
nothing, or just shut down. Disable hibernation if unwanted:

```bash
sudo systemctl mask sleep.target suspend-then-hibernate.target hibernate.target
```

---

## 7. Quick daily-driver checklist

```bash
sudo smartctl -H /dev/sda                       # PASSED?
sudo smartctl -a /dev/sda | grep -Ei 'wear_leveling|total_lbas_written|reallocated'
cat /sys/block/sda/queue/discard_granularity    # non-zero = discard OK
systemctl list-timers fstrim.timer              # weekly trim scheduled
grep -E 'discard|noatime' /etc/fstab            # no discard, has noatime
df -h /                                        # >=15% free
```

---

## 8. Field reference (cheat sheet)

| Goal | Command |
|---|---|
| NVMe wear | `sudo nvme smart-log /dev/nvme0 \| grep percentage_used` |
| SATA wear | `sudo smartctl -a /dev/sda \| grep -Ei 'wear_leveling\|total_lbas_written'` |
| Health pass | `sudo smartctl -H /dev/sda` |
| Discard supported | `cat /sys/block/sda/queue/discard_granularity` |
| Trim all now | `sudo fstrim -a -v` |
| Weekly trim on | `sudo systemctl enable --now fstrim.timer` |
| Drop continuous discard | edit `/etc/fstab`, remove `discard`, `mount -o remount /` |
| Less metadata writes | add `noatime` to fstab SSD mounts |
| Headroom for OP | keep 15–20% free / leave unpartitioned space |

---

## 9. Your System — specifics (measured 2026-07-11)

**Detected:**

| Item | Value | Note |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS, kernel 6.17.0-35 | — |
| RAM | 7.5 GiB total, 1.7 GiB free under load | **Tight** → swap gets used |
| SSD | `sda` Zebronics 2.5SSD256GB (SATA) | root `/` on `sda2`, ext4 |
| SSD usage | 234 GiB, 70% used, **68 GiB free** | Good OP headroom (>20% free) |
| HDD | `sdb` Toshiba MQ01ABD100 1 TB (spinning) | bulk data, `sdb6` 98% full |
| NVMe | none | — |
| fstrim.timer | **enabled**, runs weekly (last Mon) | Already correct — keep |
| `discard` in fstab | **no** | Already correct — keep |
| discard support | granularity 512, max ≈2.1 GB | Supported |
| scheduler | `mq-deadline` | Fine for SSD |
| smartmontools | **not installed** | You are blind to wear |
| fstab mount opts | `defaults` (relatime, no noatime) | Metadata writes on reads |
| `/tmp` | on root SSD (no tmpfs) | Temp writes burn SSD |
| swap | `/swap.img` **10 GiB on the SSD**, swappiness **60** (Ubuntu default), **2.5 GiB in use** | Keep on SSD as-is |

### 9a. Action 1 — install monitoring (closes the blind spot)

```bash
sudo apt update
sudo apt install -y smartmontools

# Baseline your wear now (re-run monthly)
sudo smartctl -a /dev/sda | grep -Ei 'wear_leveling|total_lbas_written|reallocated|power_on_hours|percentage used'

# Enable background health alerts
sudo systemctl enable --now smartd
```

### 9b. Action 2 — add `noatime` to the SSD root mount

```bash
sudoedit /etc/fstab
# change the root line from:
#   /dev/disk/by-uuid/3a7f...  /  ext4  defaults  0 1
# to:
#   /dev/disk/by-uuid/3a7f...  /  ext4  defaults,noatime  0 1
sudo mount -o remount,noatime /
```

### 9c. Action 3 — put `/tmp` and `/var/tmp` on tmpfs (RAM)

```bash
sudoedit /etc/fstab
# append:
#   tmpfs  /tmp       tmpfs  defaults,noatime,size=2G  0 0
#   tmpfs  /var/tmp   tmpfs  defaults,noatime,size=1G  0 0
sudo mount -a
```

### 9d. Action 4 — cap journald disk churn

```bash
sudoedit /etc/systemd/journald.conf
# set:
#   SystemMaxUse=500M
#   SystemKeepFree=1G
#   MaxRetentionSec=2week
sudo systemctl restart systemd-journald
```

### 9e. Keep as-is (already correct)

- **Weekly `fstrim.timer`** — verified enabled and running. Do not add `discard`
  to fstab.
- **`mq-deadline` scheduler** — appropriate for the SSD.
- **SSD 70% full / 68 GiB free** — enough unpartitioned-style headroom for the
  controller's wear-leveling; no need to shrink.

### 9f. Separate (not SSD wear) — HDD `sdb6` is 98% full

`sdb6` has only 2.2 GiB free. That's a spinning disk, so it won't wear out from
writes, but it will thrash and fragment, hurting perf and risking ENOSPC.
Offload media/cold data elsewhere; don't let it hit 100%.

### 9g. Targeted checklist for THIS machine

```bash
sudo apt install -y smartmontools && sudo systemctl enable --now smartd
sudo smartctl -a /dev/sda | grep -Ei 'wear_leveling|total_lbas_written'   # baseline
# add noatime + tmpfs /tmp,/var/tmp to /etc/fstab; mount -a
# cap journald to 500M
systemctl list-timers fstrim.timer    # confirm weekly (already on)
df -h / ; df -h "/media/harmeet/Disk: F"   # SSD >=20% free ok; HDD needs cleanup
```

---

## 10. Performance tuning — safe, targeted to THIS machine

Profile: **Intel i5-8250U** (4C/8T, turbo to 3.4 GHz, 6 MB L3), **7.5 GiB RAM**,
Ubuntu 24.04, kernel 6.17. CPU governor is **`powersave`**, I/O scheduler is
**`mq-deadline`** on both disks, **zswap disabled**, **`dirty_ratio`/`dirty_background_ratio`
report `0`** (kernel defaults), **38 snaps** installed, **`plocate-updatedb`** and
**`apt-daily`** run heavy daily I/O, one **failed unit** (`snap-firefox-8504.mount`),
**UFW active** (keep it — security beats micro-optimization).

All changes below are **reversible** and safe. Each has a `Hidden insight` note
explaining the real mechanism, not just the command.

### 10a. CPU governor → `performance` (biggest single win)

Your CPU is pinned to `powersave`, which on `intel_pstate` is conservative and
lags on turbo decisions. `performance` lets the chip actually use its 3.4 GHz
turbo and feels dramatically snappier for compile/launch/responsiveness.

```bash
# Check current
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor   # -> powersave
# Apply now (all cores)
sudo sed -i 's/^GOVERNOR=.*/GOVERNOR=performance/' /etc/initcpio/...   # n/a on Ubuntu
# Ubuntu uses cpufreq, set per-boot via systemd:
sudo apt install -y cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl restart cpufrequtils
# Or one-shot without the package:
for c in /sys/devices/system/cpu/cpu[0-7]/cpufreq/scaling_governor; do echo performance | sudo tee $c; done
```

> **Hidden insight:** on `intel_pstate`, `performance` is *not* a hard clock pin —
> it's the "let turbo happen" mode. `powersave` doesn't save meaningful battery on
> a modern Intel part but *does* cap responsiveness. If you're on battery a lot,
> `schedutil` is the best of both (kernel scheduler-driven). Install: it ships in
> `linux-tools-common`; `GOVERNOR="schedutil"`.

### 10b. SSD scheduler → `none` (lower latency); HDD stays `mq-deadline`

`none` is the no-op (no-op / NOP) elevator. Modern SSDs do their own internal
queueing, so a software scheduler like `mq-deadline` only adds latency. HDDs
still benefit from `mq-deadline`.

```bash
# Immediate (sda = SSD)
echo none | sudo tee /sys/block/sda/queue/scheduler
# Persist via udev rule
echo 'ACTION=="add|change", KERNEL=="sda", ATTR{queue/scheduler}="none"' | sudo tee /etc/udev/rules.d/60-ssd-scheduler.rules
# Verify
cat /sys/block/sda/queue/scheduler   # -> [none]
```

> **Hidden insight:** `cat /sys/block/sda/queue/scheduler` prints `none [mq-deadline]` —
> the bracket is the *active* one. The kernel default for SATA SSDs flipped to
> `mq-deadline` for safety, but NVMe has used `none` for years. For a SATA SSD the
> `none` path shaves per-request queuing overhead, most visible on many small
> random reads (app launches, package installs).

### 10c. Enable `zswap` (compress cold RAM instead of hitting the SSD swap)

`zswap` is **disabled** on your box (`/sys/module/zswap/parameters/enabled = N`).
With swap on the SSD and only 7.5 GiB RAM, enabling zswap means cold pages get
LZ4-compressed in RAM and only spill to the SSD when truly necessary — less
swap I/O, snappier under memory pressure, and (bonus) less SSD wear.

```bash
# Runtime enable
echo lz4 | sudo tee /sys/module/zswap/parameters/compressor
echo 1    | sudo tee /sys/module/zswap/parameters/enabled
# Persist across reboots (GRUB)
sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="/&zswap.enabled=1 zswap.compressor=lz4 /' /etc/default/grub
sudo update-grub
```

> **Hidden insight:** `zswap` sits *in front of* your real swap. It trades a little
> CPU (LZ4 is ~500 MB/s/core) for far fewer SSD page-outs. Because you kept the
> default `swappiness=60`, the kernel will still swap — zswap just makes that swap
> ~2–3× cheaper. Use `lz4` (fast) not `zstd` unless you want max compression at CPU cost.

### 10d. Bound buffered-write bursts — set explicit dirty ratios

`dirty_ratio`/`dirty_background_ratio` show `0`, meaning kernel defaults (20 / 10).
On a 7.5 GiB box that lets up to ~1.5 GiB of dirty pages pile up before a forced
flush — a visible "freeze then flush" stutter and a big SSD write spike.

```bash
echo 'vm.dirty_background_ratio=5'  | sudo tee -a /etc/sysctl.d/99-perf.conf
echo 'vm.dirty_ratio=15'            | sudo tee -a /etc/sysctl.d/99-perf.conf
echo 'vm.dirty_writeback_centisecs=1500' | sudo tee -a /etc/sysctl.d/99-perf.conf
sudo sysctl --system
```

> **Hidden insight:** lower values = smaller, more frequent flushes = smoother
> interactivity at the cost of slightly lower bulk-copy throughput. For a desktop
> SSD this is the right trade. If you do huge `dd`/copy jobs often, bump
> `dirty_ratio` back toward 20 for that session only.

### 10e. Tame the daily disk hogs: `plocate-updatedb` + `apt-daily`

`systemd-analyze blame` shows `apt-daily.service` (47 s) and
`plocate-updatedb.service` (41 s) doing heavy background I/O — they scan the
whole filesystem and hammer the SSD, often right when you start working.

```bash
# If you don't use `locate`, disable the DB rebuild entirely:
sudo systemctl disable --now plocate-updatedb.timer plocate-updatedb.service
# If you do use it, run it weekly + idle-IO only:
sudo systemctl enable --now plocate-updatedb.timer
sudo sed -i 's/^/#/' /etc/cron.daily/plocate 2>/dev/null
# ionice the remaining one (edit the timer's service ExecStart to prepend `ionice -c3`)
# apt-daily: stop it from auto-rebooting and throttle its window
sudo systemctl mask apt-daily-upgrade.service   # no surprise reboots
```

> **Hidden insight:** `plocate-updatedb` walks every file including the 98%-full HDD
> and the snap loop mounts — that's where the 41 s comes from. Disabling it removes
> a daily full-FS scan. `apt-daily-upgrade` masked = no forced reboots mid-work.

### 10f. Snap hygiene — kill loop-mount overhead and a failed unit

38 snaps, each mounting a squashfs loop device at boot (~2–3 s each) and holding
RAM. There's also a **failed unit** (`snap-firefox-8504.mount not-found`) from a
removed Firefox snap.

```bash
# Clear the failed-unit error state
sudo systemctl reset-failed
# List what you actually use; remove the rest
snap list
snap remove firefox beekeeper-studio   # if unused (firefox mount is already broken)
# Stop snap from auto-updating at the worst time (hold refreshes 30 days)
sudo snap refresh --hold=30d
# Optionally disable snapd's auto-import/seeding noise you don't need
sudo systemctl disable --now snapd.autoimport.service
```

> **Hidden insight:** every snap is a compressed `squashfs` mounted via a loop
> device — they consume loop devices (limited pool) and add mount latency to boot
> plus resident RAM for each app's base image. Brave/Code/Firefox as snaps means
> three extra loop mounts. If you can install those as `.deb`/native instead, you
> shed boot time and RAM. The broken `snap-firefox` mount also makes `systemctl
> --failed` always red — `reset-failed` clears the cosmetic error.

### 10g. Keep more file-metadata in RAM — lower `vfs_cache_pressure`

Default `vfs_cache_pressure=100`. With 7.5 GiB and currently **zero memory
pressure** (PSI `some`/`full` all 0.00), you can safely keep dentries/inode caches
longer so file browsing and tab-completion stay instant.

```bash
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.d/99-perf.conf
sudo sysctl --system
```

> **Hidden insight:** this only helps when RAM is plentiful and there's no memory
> contention — which is your case right now. If you later run memory-heavy jobs,
> the kernel still reclaims cache first, so it's safe; 50 just biases retention.

### 10h. Disable ModemManager only if you have no cellular/WWAN modem

**Bluetooth is kept ON** — you use it daily, so `bluetooth.service` stays enabled.
(On Intel laptops BT and Wi-Fi share one combo chip; since BT is in active use,
leave it.) Only `ModemManager` is a candidate to disable, and only if you never
tether via a cellular modem.

```bash
# Only if you have no cellular/WWAN modem:
sudo systemctl disable --now ModemManager.service
```

> **Hidden insight:** `ModemManager` polls for a modem you don't have; disabling it
> is pure overhead removal with no downside when there's no WWAN. BT itself is a
> real feature here, so it stays. Reversible.

### 10i. Faster boot — drop `splash`, keep `quiet`

GRUB has `quiet splash`. The Plymouth splash adds graphical init; removing
`splash` shaves boot time and shows real logs if something breaks.

```bash
sudo sed -i 's/ splash//g' /etc/default/grub
sudo update-grub
```

> **Hidden insight:** `splash` is cosmetic but it serializes the init path through
> Plymouth; on an SSD it's only ~1–2 s, but it also hides boot errors. Keep `quiet`.

### 10j. Raise HDD read-ahead for sequential media (HDD only)

The spinning `sdb` benefits from larger read-ahead for big sequential reads
(videos, backups). The SSD doesn't need it.

```bash
# 256 * 512B = 128 KiB read-ahead on the HDD
sudo blockdev --setra 256 /dev/sdb
# Persist (udev)
echo 'ACTION=="add|change", KERNEL=="sdb", ATTR{queue/read_ahead_kb}="128"' | sudo tee /etc/udev/rules.d/60-hdd-ra.rules
```

> **Hidden insight:** read-ahead on an SSD can *hurt* (prefetching useless sectors
> wastes bandwidth), but on a mechanical disk it masks seek latency for sequential
> streams. Target `sdb` only.

### 10k. Journald — keep the default rate-limit ON

You chose to keep the default `RateLimit` behavior, so **no change is made here**.
The 500 M size cap from §9e still applies (wear control). If you later debug a
crash loop and suspect truncated logs, you *may* temporarily widen the limit —
but the default is the safe, intended setting and stays as-is.

```bash
# No action required — RateLimit remains at distribution default.
grep -E '^RateLimit' /etc/systemd/journald.conf || echo "using compiled defaults"
```

> **Hidden insight:** journald `RateLimit` silently drops log lines during bursts
> (e.g. a crash loop) to protect the disk from a log storm. That's a *feature* for
> a desktop — leaving it on prevents a buggy service from flooding the SSD with
> writes. Only disable it deliberately while debugging.

### 10l. NUMA / IRQ — leave alone, but know why

`irqbalance` is **inactive** and that's fine: a single-socket laptop with one
NUMA node gets no benefit from IRQ balancing (modern kernels spread IRQs
themselves). Don't enable it — it's a no-op win you'd be adding complexity for.

> **Hidden insight:** multi-socket servers need `irqbalance`; a laptop with one
> CCD does not. The "always run irqbalance" advice is cargo-culted from server
> tuning. Skipping it is the correct, faster choice here.

### 10m. What NOT to touch (so you don't lose the gains)

- **UFW** — keep active; the CPU cost is negligible and the security is not.
- **`swappiness`** — you chose the Ubuntu default (60); leave it (zswap covers the
  swap-cheapening instead, §10c).
- **`mq-deadline` on the HDD** — correct for spinning media; don't switch `sdb` to `none`.
- **Transparent Huge Pages** — `madvise` is the safe default; `always` can
  fragment/regress some workloads. Leave it.
- **Overclocking / undervolting** — skip on a laptop; thermal headroom is the
  real limiter and pstate `performance` already unlocks turbo.

### 10n. Apply-in-order checklist

```bash
# 1 CPU
for c in /sys/devices/system/cpu/cpu[0-7]/cpufreq/scaling_governor; do echo performance | sudo tee $c; done
# 2 SSD scheduler
echo none | sudo tee /sys/block/sda/queue/scheduler
# 3 zswap
echo lz4 | sudo tee /sys/module/zswap/parameters/compressor; echo 1 | sudo tee /sys/module/zswap/parameters/enabled
# 4 dirty + cache sysctl
cat <<'EOF' | sudo tee -a /etc/sysctl.d/99-perf.conf
vm.dirty_background_ratio=5
vm.dirty_ratio=15
vm.dirty_writeback_centisecs=1500
vm.vfs_cache_pressure=50
EOF
sudo sysctl --system
# 5 daily hogs
sudo systemctl disable --now plocate-updatedb.timer plocate-updatedb.service
sudo systemctl mask apt-daily-upgrade.service
# 6 snap cleanup
sudo systemctl reset-failed
# 7 persist zswap/scheduler/ra in grub+udev (see 10b/10c/10j)
sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="/&zswap.enabled=1 zswap.compressor=lz4 /; s/ splash//g' /etc/default/grub
sudo update-grub
```

### 10o. Everyday impact — what you'll actually feel, and why

Each tweak below, in plain terms: the visible change in daily use, then the
mechanism that produced it.

**10a. CPU governor → `performance`**
- *What you feel:* apps open faster, builds/compiles finish sooner, the machine
  stops feeling "lazy" for the first few seconds after you click something.
  Video calls and Brave tabs stop stuttering under load.
- *Why it's faster:* `powersave` on `intel_pstate` under-clocks and is slow to
  request turbo; `performance` lets the 1.6→3.4 GHz turbo engage immediately.
  It's not pinning the clock high — it just removes the lazy ramp, so the CPU is
  at the right frequency when you actually need it.

**10b. SSD scheduler → `none`**
- *What you feel:* snappier app launches, faster `apt`/package installs, quicker
  searches over many small files. Less "think about it" pause before disk-backed
  actions complete.
- *Why it's faster:* your SSD already queues and reorders I/O internally. A
  software elevator (`mq-deadline`) adds a scheduling step the hardware does
  better itself. Dropping it removes per-request latency — most noticeable on
  lots of small random reads, which is exactly what launching apps does.

**10c. Enable `zswap`**
- *What you feel:* the system stays responsive when RAM fills (many Brave tabs,
  Docker, opencode running). Fewer "everything froze for a second" moments; the
  fan/disk don't spin up as hard during memory pressure.
- *Why it's faster:* instead of writing cold memory pages straight to the SSD
  swapfile (slow, and wears the drive), the kernel LZ4-compresses them in RAM
  first. Compressed pages are ~2–3× smaller, so far less reaches the SSD. With
  your 7.5 GiB and swap on the SSD, this is the single best swap-cheapening lever
  (you kept `swappiness=60`, so swap *will* happen — zswap makes it cheap).

**10d. Bound dirty ratios**
- *What you feel:* no more "everything pauses, then the disk light goes crazy"
  after a big copy or save. The UI stays interactive during large writes.
- *Why it's faster:* by default up to ~1.5 GiB of written data can sit in RAM
  before the kernel force-flushes it all at once — that burst is the freeze.
  Capping `dirty_background_ratio=5` / `dirty_ratio=15` makes the kernel trickle
  writes to the SSD in smaller, frequent batches, so the flush never becomes a
  visible wall. Trade-off: bulk `dd`/copy throughput drops slightly; worth it
  for desktop feel.

**10e. Tame `plocate-updatedb` + `apt-daily`**
- *What you feel:* the machine no longer "gets slow around the same time every
  day" for no reason, and you won't get surprise reboots. Disk stays quiet when
  you're actually using it.
- *Why it's faster:* these scan the *entire* filesystem (including the 98%-full
  HDD and every snap loop mount) daily, saturating I/O for 40+ seconds. That I/O
  competes with whatever you're doing. Disabling `plocate` removes a full-FS walk;
  masking `apt-daily-upgrade` stops forced reboots. The CPU time is secondary to
  the I/O contention you avoid.

**10f. Snap hygiene**
- *What you feel:* faster boot, a bit more free RAM, and `systemctl --failed`
  stops showing a scary red error. Fewer background snap-refresh spikes.
- *Why it's faster:* every snap is a squashfs mounted through a loop device —
  boot waits on those mounts, and each app's base image sits in RAM. 38 snaps =
  38+ loop mounts and resident image memory. Removing unused ones (and the broken
  `snap-firefox` mount) frees loop devices, RAM, and boot seconds. `reset-failed`
  just clears the stale error state.

**10g. Lower `vfs_cache_pressure` → 50**
- *What you feel:* file browsing, `ls`, tab-completion, and opening recent files
  stay instant even after the machine has been idle or under load.
- *Why it's faster:* this cache holds directory/inode metadata. At `100` the
  kernel reclaims it aggressively; at `50` it keeps more of it in RAM. Since your
  PSI memory pressure is 0.00 (no contention), hoarding this metadata means the
  kernel answers "where is this file?" from RAM instead of re-reading the disk.
  Safe because RAM is plentiful and the kernel still reclaims cache first under
  real pressure.

**10h. ModemManager only (Bluetooth kept ON)**
- *What you feel:* (only if you disabled ModemManager) a hair less idle CPU; no
  change if you kept it. Bluetooth stays fully working.
- *Why it (might) help:* `ModemManager` polls for a cellular modem you don't own,
  pure overhead. Bluetooth is a feature you use, so it's left enabled — disabling
  it would cost you functionality for a negligible gain.

**10i. Drop `splash`**
- *What you feel:* boot finishes a touch quicker and, if something breaks, you
  see the real error instead of a spinner.
- *Why it's faster:* Plymouth serializes part of init through a graphical splash.
  Removing it cuts that path. Minor in seconds, but it also removes a failure
  mode where a hung splash masks the actual problem.

**10j. HDD read-ahead bump (HDD only)**
- *What you feel:* smoother playback/scrubbing of large video files and faster
  big sequential copies *from the HDD*; no change (and no harm) on the SSD.
- *Why it's faster:* a spinning disk pays a seek penalty per read; fetching a
  larger sequential chunk upfront hides that latency for streaming/backup work.
  Deliberately scoped to `sdb` only — on the SSD, prefetching wastes bandwidth.

**10k. Journald rate-limit kept ON (default)**
- *What you feel:* nothing day-to-day; the disk is protected from a runaway
  log storm.
- *Why it matters:* the default `RateLimit` drops burst lines to shield the SSD
  from a logging flood — a good thing on a desktop. Left at default as you chose;
  only widen it temporarily when actively debugging.

**10l. Leave `irqbalance` off**
- *What you feel:* nothing — and that's correct.
- *Why it's right:* a single-socket laptop has one NUMA node; the kernel already
  spreads IRQs well. Enabling `irqbalance` would add a daemon and complexity for
  zero benefit. Not touching it *is* the optimized choice here.

**The "don't touch" list, in daily terms**
- *UFW*: you'd trade a trivial CPU sliver for real attack surface — not worth it.
- *swappiness=60*: you chose the Ubuntu default; zswap (10c) already makes swap
  cheap, so no need to fight the knob.
- *HDD `mq-deadline`*: switching the spinner to `none` would *hurt* it (no internal
  queue to lean on) — keep it.
- *THP `madvise`*: `always` can fragment memory and regress some apps; the default
  is the safe, fast choice.

**Net everyday result:** with 10a–10k applied you should notice a machine that
boots a few seconds quicker, launches apps and installs packages snappily, stays
responsive when RAM fills, and never randomly "gets slow" mid-day from background
scans — all without sacrificing the SSD's lifespan (§9) or security (UFW on).

