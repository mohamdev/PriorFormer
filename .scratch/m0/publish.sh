#!/usr/bin/env bash
# Publish the M0 tickets in issues/ as GitHub issues, in dependency order.
# Needs gh installed and logged in (gh auth login). Run from anywhere.
set -euo pipefail
REPO=mohamdev/PriorFormer
DIR="$(cd "$(dirname "$0")" && pwd)/issues"

gh label create ready-for-agent --repo "$REPO" --color 0E8A16 --description "Ready for an agent to pick up" 2>/dev/null || true
gh label create ready-for-human --repo "$REPO" --color D93F0B --description "Needs a human (e.g. cloud runs)" 2>/dev/null || true

declare -A NUM   # ticket number (01..) -> GitHub issue number
for f in "$DIR"/[0-9][0-9]-*.md; do
  id=$(basename "$f" | cut -c1-2)
  title=$(head -1 "$f" | sed 's/^# //')
  label=$(sed -n 's/^<!-- label: \(.*\) -->$/\1/p' "$f")
  blockers=$(sed -n 's/^<!-- blocked-by: \(.*\) -->$/\1/p' "$f" | tr -d ' ')
  body=$(tail -n +2 "$f" | grep -v '^<!-- ')
  body+=$'\n\n## Blocked by\n\n'
  if [ -z "$blockers" ]; then
    body+="None (can start immediately)"
  else
    for b in ${blockers//,/ }; do body+="- #${NUM[$b]}"$'\n'; done
  fi
  url=$(gh issue create --repo "$REPO" --title "$title" --label "$label" --body "$body")
  NUM[$id]=${url##*/}
  echo "$id -> $url"
done
