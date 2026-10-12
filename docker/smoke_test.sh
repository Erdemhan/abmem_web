#!/bin/sh
# End-to-end smoke test of a running ABMEM docker compose stack.
#
# Checks that the app answers, the pages and static files load, the starter
# data was imported, and one starter scenario runs to FINISHED with its
# analysis page and exports available.
#
# Usage (from the folder with docker-compose.yml, stack already started):
#   sh docker/smoke_test.sh [base_url] [compose_project]
# Defaults: http://127.0.0.1:8000 and the default compose project name.
set -eu

BASE="${1:-http://127.0.0.1:8000}"
PROJECT="${2:-}"
SCENARIO="${SMOKE_SCENARIO:-Only Renewable}"
RUN_TIMEOUT="${SMOKE_RUN_TIMEOUT:-600}"

compose() {
    if [ -n "$PROJECT" ]; then docker compose -p "$PROJECT" "$@"; else docker compose "$@"; fi
}
manage_shell() {
    compose exec -T web python manage.py shell -c "$1" 2>/dev/null | tail -n 1
}
fail() { echo "FAIL: $*" >&2; exit 1; }
ok() { echo "ok   $*"; }
status_of() { curl -s -o /dev/null -w "%{http_code}" "$1"; }

echo "Waiting for $BASE ..."
i=0
until [ "$(status_of "$BASE/dev/")" = "200" ]; do
    i=$((i + 1))
    [ "$i" -ge 100 ] && fail "the app did not answer within 300 s"
    sleep 3
done
ok "app answers"

[ "$(status_of "$BASE/")" = "302" ] || fail "/ does not redirect"
for path in /dev/ /dev/agent/list /dev/resource/list /dev/simulation/list \
            /dev/simulation/list-proxy /dev/demand-profiles/ /dev/algorithms/ /dev/hpo/ \
            "/static/assets/css/abmem.css?v=3" /static/assets/js/plugins/chartjs.min.js; do
    code=$(status_of "$BASE$path")
    [ "$code" = "200" ] || fail "$path returned $code"
done
ok "pages and static files"

counts=$(manage_shell "from abmem.models import Resource, Simulation; print(Resource.objects.count(), Simulation.objects.filter(proxy=True).count())")
set -- $counts
[ "${1:-0}" -gt 0 ] && [ "${2:-0}" -gt 0 ] || fail "starter data missing (resources, scenarios: $counts)"
ok "starter data: $1 resources, $2 scenarios"

scenario_id=$(manage_shell "from abmem.models import Simulation; s = Simulation.objects.filter(proxy=True, name='$SCENARIO').first(); print(s.id if s else '')")
[ -n "$scenario_id" ] || fail "scenario '$SCENARIO' not found"

response=$(curl -s -H "Accept: application/json" "$BASE/dev/simulation/run/$scenario_id/")
run_id=$(echo "$response" | sed -n 's/.*"run_id": *\([0-9][0-9]*\).*/\1/p')
[ -n "$run_id" ] || fail "run did not start: $response"
ok "run $run_id started from scenario '$SCENARIO'"

start=$(date +%s)
while :; do
    json=$(curl -s "$BASE/dev/simulation/status/?simulation_id=$run_id")
    state=$(echo "$json" | sed -n 's/.*"status": *"\([A-Z]*\)".*/\1/p')
    case "$state" in
        FINISHED) break ;;
        FAILED|CANCELED|CANCELLED) fail "run $run_id ended as $state: $(echo "$json" | sed -n 's/.*"error": *\("[^"]*"\).*/\1/p')" ;;
    esac
    [ $(( $(date +%s) - start )) -ge "$RUN_TIMEOUT" ] && fail "run $run_id not finished after $RUN_TIMEOUT s (state $state)"
    sleep 5
done
ok "run $run_id FINISHED in $(( $(date +%s) - start )) s"

for path in "/dev/simulation/$run_id/analysis/" "/dev/simulation/$run_id/analysis/data/" \
            "/dev/simulation/$run_id/analysis/offers/" "/dev/simulation/$run_id/export/offers.csv" \
            "/dev/simulation/$run_id/export/parameters.json"; do
    code=$(status_of "$BASE$path")
    [ "$code" = "200" ] || fail "$path returned $code"
done
ok "analysis page and exports"

files=$(compose exec -T web sh -c 'ls /app/sim_data | wc -l')
[ "$files" -gt 0 ] || fail "no JSON offer export in /app/sim_data"
ok "JSON offer export written"

echo "SMOKE TEST PASSED"
