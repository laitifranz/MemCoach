#!/usr/bin/env bash

# Mapping from logical cluster env names to space-separated aliases.
# Behaviour:
#   - For **hostname auto-detection**, each token in the value is matched
#     against the current hostname; if any token is contained in the hostname,
#     the corresponding key is returned as the env (e.g. host 'disi-yoda'
#     matches alias 'yoda' → env 'todi').
#   - For **explicit --cluster <value>**, if <value> is not a key in this
#     map, we search all alias tokens; if a token equals <value>, we map it
#     back to the owning key. This allows things like:
#       --cluster leo  → env 'leonardo'
declare -A CLUSTER_ALIASES=(
  ["leonardo"]="leonardo leo"
  ["todi"]="yoda todi"
)

detect_cluster_env() {
  local host
  host="$(hostname 2>/dev/null || echo "${HOSTNAME:-}")"

  local env aliases alias
  for env in "${!CLUSTER_ALIASES[@]}"; do
    aliases="${CLUSTER_ALIASES[$env]}"
    for alias in $aliases; do
      if [[ -n "$alias" && "$host" == *"$alias"* ]]; then
        echo "$env"
        return 0
      fi
    done
  done

  return 1
}

