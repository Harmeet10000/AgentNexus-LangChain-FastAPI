#!/usr/bin/env bash
set -euo pipefail

readonly snap_name="code"
readonly user_snap_data="${HOME:?HOME must be set}/snap/code"
readonly system_snap_data="/var/snap/code"

dry_run=false
case "${1:-}" in
  "")
    ;;
  --dry-run)
    dry_run=true
    ;;
  --help)
    printf 'Usage: %s [--dry-run]\n' "$0"
    exit 0
    ;;
  *)
    printf 'Unknown option: %s\nUsage: %s [--dry-run]\n' "$1" "$0" >&2
    exit 2
    ;;
esac

if (( EUID == 0 )); then
  printf 'Run this as your normal desktop user; sudo is requested only when needed.\n' >&2
  exit 1
fi

if ! command -v snap >/dev/null 2>&1; then
  printf 'The snap command is unavailable; cannot verify or remove the Snap package.\n' >&2
  exit 1
fi

snap_installed=false
if snap list "$snap_name" >/dev/null 2>&1; then
  snap_installed=true
fi

mapfile -t snapshot_ids < <(
  snap saved "$snap_name" 2>/dev/null |
    awk -v name="$snap_name" 'NR > 1 && $2 == name && $1 ~ /^[0-9]+$/ { print $1 }' |
    sort -u
)

has_user_data=false
if [[ -e "$user_snap_data" || -L "$user_snap_data" ]]; then
  has_user_data=true
fi

has_system_data=false
if [[ -e "$system_snap_data" || -L "$system_snap_data" ]]; then
  has_system_data=true
fi

printf 'Snap-based VS Code cleanup\n\n'
printf 'Close all VS Code windows and save your work before continuing.\n\n'
printf 'Will remove:\n'
if [[ "$snap_installed" == true ]]; then
  printf '  Snap package: %s\n' "$snap_name"
fi
if ((${#snapshot_ids[@]} > 0)); then
  printf '  Saved Snap data for code, snapshot set(s): %s\n' "${snapshot_ids[*]}"
fi
if [[ "$has_user_data" == true ]]; then
  printf '  Snap-only user data: %s\n' "$user_snap_data"
fi
if [[ "$has_system_data" == true ]]; then
  printf '  Snap-only system data: %s\n' "$system_snap_data"
fi
printf '  Snap-managed revisions and desktop integration (handled by snap remove)\n'

printf '\nWill preserve:\n'
printf '  Shared settings, extensions, and sign-in: %s and %s\n' \
  "$HOME/.config/Code" "$HOME/.vscode"
printf '  Any non-Snap VS Code installation\n'

if [[ "$dry_run" == true ]]; then
  printf '\nDry run only; nothing was changed.\n'
  exit 0
fi

if [[ ! -r /dev/tty ]]; then
  printf 'Run this from an interactive terminal so the deletion can be confirmed.\n' >&2
  exit 1
fi

printf '\nThis permanently deletes the listed Snap data. Type REMOVE to continue: '
if ! IFS= read -r confirmation </dev/tty || [[ "$confirmation" != "REMOVE" ]]; then
  printf 'Cancelled; nothing was changed.\n'
  exit 1
fi

if [[ "$snap_installed" == true ]] || ((${#snapshot_ids[@]} > 0)) || [[ "$has_system_data" == true ]]; then
  sudo -v
fi

if [[ "$snap_installed" == true ]]; then
  sudo snap remove --purge "$snap_name"
  if snap list "$snap_name" >/dev/null 2>&1; then
    printf 'Snap still reports %s as installed; stopping before removing residual data.\n' \
      "$snap_name" >&2
    exit 1
  fi
fi

for snapshot_id in "${snapshot_ids[@]}"; do
  sudo snap forget "$snapshot_id" "$snap_name"
done

if [[ "$has_user_data" == true ]]; then
  rm -rf -- "$user_snap_data"
fi

if [[ "$has_system_data" == true ]]; then
  sudo rm -rf -- "$system_snap_data"
fi

if [[ -e "$user_snap_data" || -L "$user_snap_data" ]] ||
  [[ -e "$system_snap_data" || -L "$system_snap_data" ]] ||
  snap list "$snap_name" >/dev/null 2>&1; then
  printf 'Cleanup is incomplete; inspect the listed Snap paths and rerun if appropriate.\n' >&2
  exit 1
fi

printf '\nSnap-based VS Code and its Snap-only data were removed.\n'
printf 'The shared VS Code profile was preserved for the .deb installation.\n'
