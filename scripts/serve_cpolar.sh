#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CPOLAR_BIN="${CPOLAR_BIN:-${PROJECT_ROOT}/.runtime/cpolar/cpolar}"
CPOLAR_HOME="${CPOLAR_HOME:-${PROJECT_ROOT}/.runtime/cpolar/home}"
LOCAL_PORT="${MINIQWEN_WEB_PORT:-7860}"

if [[ ! -x "${CPOLAR_BIN}" ]]; then
  echo "cpolar client not found: ${CPOLAR_BIN}" >&2
  exit 2
fi

if [[ -z "${CPOLAR_AUTHTOKEN:-}" ]]; then
  if [[ -t 0 ]]; then
    read -rsp 'Cpolar authtoken: ' CPOLAR_AUTHTOKEN
    echo
  else
    echo "Set CPOLAR_AUTHTOKEN before starting the tunnel" >&2
    exit 2
  fi
fi

mkdir -p "${CPOLAR_HOME}"
chmod 700 "${CPOLAR_HOME}"
export HOME="${CPOLAR_HOME}"

"${CPOLAR_BIN}" authtoken "${CPOLAR_AUTHTOKEN}"
unset CPOLAR_AUTHTOKEN
if [[ -f "${CPOLAR_HOME}/.cpolar/cpolar.yml" ]]; then
  chmod 600 "${CPOLAR_HOME}/.cpolar/cpolar.yml"
fi

echo "Forwarding http://127.0.0.1:${LOCAL_PORT} through cpolar region ${CPOLAR_REGION:-cn}"
exec "${CPOLAR_BIN}" http -region="${CPOLAR_REGION:-cn}" "${LOCAL_PORT}"
