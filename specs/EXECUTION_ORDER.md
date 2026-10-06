# Execution order — parallel waves

**Status:** Draft, pending explicit user approval.
**Inputs:** `PLAN.md`, `ARCHITECTURE.md`, `specs/*.json`.

Specs inside one wave touch disjoint files and can be implemented by independent worker agents at the same time. A wave starts only when the previous wave is merged and green. Faz 0 runs alone first. 2.2 (`resource_filter`) runs in wave 1, in parallel with the first Faz 1 specs, as PLAN.md section 1 allows.

PLAN.md section 1 draws Faz 1 as a linear chain (1.1 -> 1.2 -> ... -> 1.14). The waves below keep every technical dependency of that chain (data, interface or file). Items without such a dependency run in parallel. Every Type B/C spec still gets its own commit with a `BUG-IMPACT` note, and the `research-debug` skill is invoked before each of them. If the user prefers the literal sequence, the same specs can be executed one by one in the order listed below.

---

## 1. Overview

```mermaid
flowchart LR
    W0["Wave 0<br/>faz0_infra"] --> W1
    subgraph W1["Wave 1 (6 workers)"]
        A1[websocket_removal]
        A2[resource_filter]
        A3[market_clearing]
        A4[observation_reward]
        A5[rl_core]
        A6[hpo.abc]
    end
    W1 --> M1["merge: 0005_merge_wave1"] --> W2
    subgraph W2["Wave 2 (8 workers)"]
        B1[rl_algorithms.ddpg]
        B2[rl_algorithms.td3]
        B3[rl_algorithms.sac]
        B4[rl_algorithms.ppo]
        B5[rl_algorithms.rdpg]
        B6[rl_algorithms.baselines]
        B7[data_model]
        B8[failstack]
    end
    W2 --> W3
    subgraph W3["Wave 3 (2 workers)"]
        C1[runtime_core]
        C2[agent_library.service]
    end
    W3 --> W4
    subgraph W4["Wave 4 (3 workers)"]
        D1[algorithm_selection_ui]
        D2[hpo.service]
        D3[validation.smoke]
    end
    W4 --> W5["Wave 5<br/>agent_library.ui"] --> W6["Wave 6<br/>hpo.ui"] --> W7["Wave 7<br/>validation.e2e (Faz 3)"]
```

| Wave | Specs (work packages) | PLAN items | Workers |
|---|---|---|---|
| 0 | `faz0_infra` | 0.1–0.7 | 1 |
| 1 | `websocket_removal`, `resource_filter`, `market_clearing`, `observation_reward`, `rl_core`, `hpo` (package `abc`) | 1.13, 2.2, 1.7, 1.1/1.3/1.4 (pure parts), 1.4/1.8/1.9 (algorithm side), 1.11 (optimizer) | 6 |
| 2 | `rl_algorithms` (6 packages), `data_model`, `failstack` | 1.4, 1.5, 1.6, 1.10 (baselines), schema for 1.2/1.10/2.1/2.3/2.4 | 8 |
| 3 | `runtime_core`, `agent_library` (package `service`) | 1.1–1.5 wiring, 1.8, 1.9, 1.10, 1.12, 2.1 service, 2.3 service | 2 |
| 4 | `algorithm_selection_ui`, `hpo` (package `service`), `validation` (package `smoke`) | 2.1 UI, 1.11 + 2.4 service, 1.14 | 3 |
| 5 | `agent_library` (package `ui`) | 2.3 UI | 1 |
| 6 | `hpo` (package `ui`) | 2.4 UI | 1 |
| 7 | `validation` (package `e2e`) | Faz 3 | 1 |

PLAN dependency check: 2.1 after 1.8/1.9 (wave 4 after wave 3), 2.3 after 2.1 and 2.2 (wave 5 after waves 1 and 4), 2.4 after 1.8, 1.10, 1.11 and 2.3 (wave 6 after waves 3, 4 and 5), Faz 3 last.

---

## 2. File ownership per wave

Paths are relative to the repository root. "new" = created, "del" = deleted. A file listed for two specs of the same wave would be a conflict. None exists below.

### Wave 0 — `faz0_infra` (single worker)
`.gitignore`, `requirements.txt` (new), `.env.example` (new), `README.md`, `pytest.ini` (new), `conftest.py` (new), `abmem/bootstrap.py` (new), `web_project/settings.py`, `web_project/asgi.py`, `manage.py`, `abmem/constants/__init__.py`, `abmem/constants/paths.py`, `abmem/services/__init__.py`, `abmem/services/agent/__init__.py`, `abmem/models/{agent,market,simulation}.py` (header lines only), every module with `sys.path.append` or import-time `django.setup()` (listed in the spec), `abmem/services/algorithms/optimization/{hyperparam_optimization,moabc,optimizer}.py` (imports only), `abmem/services/file_reader/reader_service.py` (paths only), `tests/**` (new), `.claude/context/FORMULATION.md` (new, outline only).

Faz 0 also pre-creates every test package so that parallel workers never create the same `__init__.py`: `tests/{characterization,market,agent,algorithms,models,runtime,views,hpo,smoke,e2e,perf}/__init__.py`. Workers skip any `tests/<dir>/__init__.py` that their spec lists.

### Wave 1
| Spec | Files |
|---|---|
| `websocket_removal` | `abmem/urls.py`, `web_project/asgi.py`, `web_project/settings.py`, `requirements.txt`, `README.md`, `abmem/consumers.py` (del), `abmem/routing.py` (del), `tests/test_no_websocket.py` (new) |
| `resource_filter` | `abmem/services/agent/resource_rules.py` (new), `abmem/forms.py`, `abmem/views.py`, `abmem/templates/pages/simulation-create.html`, `abmem/templates/pages/simulation-update.html`, `abmem/templates/pages/agent-list.html`, `abmem/static/assets/components/agent-form.html`, `abmem/migrations/0004_resource_ccgt_fossil.py` (new), `abmem/management/**` (new), `tests/test_resource_rules.py`, `tests/test_views_resource_filter.py` (new) |
| `market_clearing` | `abmem/services/market/clearing.py` (new), `abmem/services/market/demand.py` (new), `abmem/models/models.py`, `abmem/services/market/market_service.py`, `abmem/services/simulation/simulation_service.py`, `abmem/services/file_reader/reader_service.py`, `abmem/migrations/0004_offer_period_decimal.py` (new), `tests/market/test_clearing.py`, `tests/market/test_demand.py` (new), `tests/characterization/test_clearing_current.py`, `tests/characterization/test_period_creation_current.py` |
| `observation_reward` | `abmem/services/agent/{observation,action_space,reward}.py` (new), `abmem/services/market/accounting.py` (new), `tests/agent/test_{observation,action_space,reward}.py`, `tests/market/test_accounting.py` (new) |
| `rl_core` | `abmem/services/algorithms/{base,algorithm_config,replay_buffer,networks,registry,device}.py` (new), `abmem/services/algorithms/__init__.py`, `abmem/services/algorithms/agent_algorithm.py` (transitional lazy imports only), `abmem/models/enums.py`, `abmem/constants/config.py` (del), `tests/algorithms/{conftest,contract,test_algorithm_config,test_replay_buffer,test_networks,test_registry}.py` (new) |
| `hpo.abc` | `abmem/services/hpo/{__init__,param_space,abc_optimizer,pareto}.py` (new), `tests/hpo/test_{param_space,abc_optimizer,pareto}.py` (new) |

Wave-1 merge step (integrator): both `0004_*` migrations depend on `0003_remove_agent_seed`. After merging both branches, run `python manage.py makemigrations --merge --noinput` and rename the result to `0005_merge_wave1.py`. Then `makemigrations --check --dry-run` must report no changes.

### Wave 2
| Spec | Files |
|---|---|
| `rl_algorithms.ddpg` | `abmem/services/algorithms/DDPG.py`, `tests/algorithms/test_ddpg.py` (new) |
| `rl_algorithms.td3` | `abmem/services/algorithms/TD3.py`, `tests/algorithms/test_td3.py` (new) |
| `rl_algorithms.sac` | `abmem/services/algorithms/SAC.py`, `abmem/services/algorithms/sac_networks.py`, `tests/algorithms/test_sac.py` (new) |
| `rl_algorithms.ppo` | `abmem/services/algorithms/ppo.py`, `tests/algorithms/test_ppo.py` (new) |
| `rl_algorithms.rdpg` | `abmem/services/algorithms/RDPG.py`, `tests/algorithms/test_rdpg.py` (new) |
| `rl_algorithms.baselines` | `abmem/services/algorithms/baselines.py` (new), `tests/algorithms/test_baselines.py` (new) |
| `data_model` | `abmem/models/{agent,simulation,enums,__init__}.py`, `abmem/models/{diagnostics,hpo}.py` (new), `abmem/admin.py`, `abmem/migrations/0006…0011` (new, linear chain), `tests/models/*` (new) |
| `failstack` | `abmem/services/agent/failstack.py` (new), `tests/agent/test_failstack.py` (new) |

`registry.py` is not edited in wave 2 because `rl_core` pre-declares all class paths. The wave-2 integrator removes the `xfail` markers in `tests/algorithms/test_registry.py` after all six algorithm packages are merged. Because `rl_core` moved the algorithm imports of `agent_algorithm.py` into function bodies, rewriting the algorithm files does not break `import abmem.views` and the wave-1 view tests keep passing. The old web run path stays broken until wave 3, as it already is today (A2).

### Wave 3
| Spec | Files |
|---|---|
| `runtime_core` | `abmem/services/agent/agent_service.py`, `abmem/services/market/market_service.py`, `abmem/services/simulation/simulation_service.py`, `abmem/services/simulation/{context,seeding}.py` (new), `abmem/services/diagnostics/**` (new), `abmem/services/starter/starter_service.py`, `abmem/services/agent/portfolio_factory.py`, `abmem/services/file_reader/reader_service.py`, `abmem/views.py` (`start` view only), `abmem/urls.py` (`start` route only), deletions listed in the spec (including `agent_algorithm.py`, `actor_critic.py`, `algorithm_utils.py`, `serial_service.py`, `parallel_service.py`, `MPIP.py`, `visualization/`, `main.py`, YAML start path), `tests/runtime/*`, `tests/perf/*` (new) |
| `agent_library.service` | `abmem/services/agent/library_service.py` (new), `tests/agent/test_library_service.py` (new) |

### Wave 4
| Spec | Files |
|---|---|
| `algorithm_selection_ui` | `abmem/forms.py`, `abmem/views.py`, `abmem/templates/pages/{simulation-create,simulation-update,agent-list,watch_simulation}.html`, `abmem/static/assets/components/agent-form.html`, `abmem/templatetags/**` (new), `tests/views/test_algorithm_selection.py` (new) |
| `hpo.service` | `abmem/services/hpo/{hpo_service,worker,objective,retention}.py` (new), `abmem/management/commands/run_hpo.py` (new), `abmem/hpo_urls.py` (new, empty `urlpatterns`), `abmem/urls.py` (one include line), `abmem/bootstrap.py` (add `run_hpo_process`), `abmem/services/algorithms/optimization/optimizer.py`, deletion of `hyperparam_optimization.py`, `moabc.py`, `abc_logger.py`, `tests/hpo/test_{objective,hpo_service,retention}.py` (new) |
| `validation.smoke` | `tests/smoke/*` (new) |

### Wave 5 — `agent_library.ui`
`abmem/views.py`, `abmem/urls.py`, `abmem/forms.py`, `abmem/templates/pages/{agent-list,simulation-create,simulation-update}.html`, `abmem/static/assets/components/agent-form.html`, `tests/views/test_agent_library_views.py`, `tests/views/test_scenario_library_flow.py` (new).

### Wave 6 — `hpo.ui`
`abmem/views_hpo.py` (new), `abmem/forms_hpo.py` (new), `abmem/hpo_urls.py`, `abmem/templates/pages/hpo-{list,create,watch}.html` (new), `abmem/static/assets/components/sidebar.html`, `abmem/views.py` (`simulation_list` and `dashboard` filters), `tests/views/test_hpo_views.py` (new). If OQ-9 is approved: `abmem/models/agent.py` and/or a new model, `abmem/services/algorithms/registry.py` (parameter resolution), next free migration.

### Wave 7 — `validation.e2e`
`tests/e2e/*` (new), `experiments/*` (new), `abmem/management/commands/run_experiment.py` (new), `analiz.md`, `ARCHITECTURE.md`.

---

## 3. Shared files that force serialisation

| File | Waves that edit it (spec) |
|---|---|
| `abmem/views.py` | 1 (`resource_filter`), 3 (`runtime_core`, `start` only), 4 (`algorithm_selection_ui`), 5 (`agent_library.ui`), 6 (`hpo.ui`) |
| `abmem/urls.py` | 1 (`websocket_removal`), 3 (`runtime_core`), 4 (`hpo.service`), 5 (`agent_library.ui`) |
| `abmem/forms.py` | 1 (`resource_filter`), 4 (`algorithm_selection_ui`), 5 (`agent_library.ui`) |
| `simulation-create.html`, `simulation-update.html`, `agent-form.html`, `agent-list.html` | 1, 4, 5 |
| `abmem/services/agent/agent_service.py` | 0 (header), 3 (`runtime_core`) |
| `abmem/services/algorithms/agent_algorithm.py` | 0 (none), 1 (`rl_core`, transitional imports), 3 (`runtime_core`, delete) |
| `abmem/services/market/market_service.py` | 1 (`market_clearing`), 3 (`runtime_core`) |
| `abmem/services/simulation/simulation_service.py` | 0 (header), 1 (`market_clearing`), 3 (`runtime_core`) |
| `abmem/models/agent.py` | 0 (header), 2 (`data_model`, incl. algorithm_params per OQ-9) |
| `abmem/models/enums.py` | 1 (`rl_core`), 2 (`data_model`) |
| `abmem/models/models.py` | 1 (`market_clearing`) |
| `abmem/migrations/` | 1 (two parallel `0004_*` + merge), 2 (`0006`–`0012`) |
| `abmem/services/file_reader/reader_service.py` | 0, 1 (`market_clearing`), 3 (`runtime_core`) |
| `abmem/bootstrap.py` | 0, 4 (`hpo.service`) |
| `web_project/settings.py`, `requirements.txt` | 0, 1 (`websocket_removal`) |

---

## 4. Gating open questions per wave

**Status 2026-10-06: all open questions OQ-1 to OQ-19 are resolved by the user (ARCHITECTURE.md section 11). No wave is blocked.** Every spec carries a binding `resolved_open_questions` block. The table below is kept as a record of which wave depended on which decision.

Consequences of the resolutions for the wave plan:
- Wave 2 (`data_model`) adds every resolved schema item in one linear chain: `0010_simulation_fields_agentsteplog` (seed, observation_lags, evaluation_periods, error, AgentStepLog), `0011_hpo` (with heartbeat_at), `0012_algorithm_params` (Agent.algorithm_params, AlgorithmDefault). Waves 3 and 6 add no migration.
- Wave 4 smoke tests use 60 periods and a test-only `batch_size = learning_starts = 8` override for learning-path checks (OQ-7).

| Wave | Hard-blocking | Soft-blocking (proposal used as default) |
|---|---|---|
| 0 | OQ-10 (FORMULATION.md location) | none |
| 1 | OQ-18 (equal-price split rule, `market_clearing`), OQ-12 (zero demand) | OQ-3, OQ-4, OQ-6 (`rl_core` defaults), OQ-2 (input array only), OQ-19 (`hpo.abc` batched onlookers) |
| 2 | OQ-15 (migration `0008` only), OQ-13 (`failstack`) | OQ-4, OQ-6, OQ-5/OQ-8/OQ-9/OQ-17 fields are added only if already answered |
| 3 | OQ-1, OQ-2, OQ-8 | OQ-5, OQ-7 (test sizes) |
| 4 | OQ-11 (`hpo.service` objective) | OQ-7, OQ-19 |
| 5 | OQ-14 | none |
| 6 | OQ-9 (apply button only) | OQ-17 |
| 7 | OQ-9 (last journey step) | none |

---

## 5. Per-wave protocol

1. The integrator creates the wave base commit. Each worker works in its own git worktree and branch from that base (`git worktree add ../abm_wave<N>_<spec> -b wave<N>/<spec>`), so concurrent test runs do not see each other's unfinished files.
2. A worker receives only its spec JSON (or work package), the referenced interface signatures and the relevant line ranges, following the context isolation rule.
3. The worker implements, writes tests, runs `pytest` for its own test files plus `tests/characterization`, checks `.gitignore` with `git check-ignore` before staging, and commits. Type B/C specs: `research-debug` skill first, `BUG-IMPACT` in the commit message.
4. The integrator merges branches in the order of the wave table, runs the full `pytest`, `python manage.py check` and `python manage.py makemigrations --check --dry-run`, then re-indexes the code graph (`index_repository`) and records ADR changes.
5. `@integration-verifier` checks the cross-spec contracts of the wave (interfaces in ARCHITECTURE.md section 5) before the next wave starts.
