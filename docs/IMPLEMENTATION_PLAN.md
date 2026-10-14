# Implementation Plan

Companion to [`ARCHITECTURE.md`](../ARCHITECTURE.md), the design spec. This document covers:

- what is built, phase by phase;
- spec gaps found while planning, and how each will be resolved;
- the phased ticket list, with acceptance criteria for every phase.

Phases are gated by their **exit criteria, not by the calendar**. The spec's §13 timeline (8–9 weeks) is optimistic for one developer. The order is fixed, the dates are not.

| Phase | Theme | Status |
|---|---|---|
| 0 | Project scaffold + data ingestion | Planned |
| 1 | Graph store + DAG editor (frontend) | Planned |
| 2 | Identification + estimation API | Planned |
| 3 | Refutation pipeline | Planned |
| 4 | NOTEARS auto-discovery | Planned |
| 5 | Results dashboard + polish | Planned |
| 6 | Testing, validation, documentation | Planned |

---

## 1. Spec gaps and resolutions

These critical domain and architectural findings govern the implementation:

1. **Column type auto-detection vs. manual declaration (§6.1).** The spec requires column types at ingestion, but users may not know types for all 35 columns in IBM HR.
   - *Resolution:* Implement **auto-detection with override**. Continuous: numeric with > 10 unique values. Binary: 2 unique values. Ordinal: integer with 3–10 unique values. Categorical: everything else. Users can override any auto-detected type in the UI before proceeding.

2. **Graph store architecture & Apache AGE overkill (§7.2, ADR-2).** Apache AGE brings heavy Docker C-extension debt, Cypher debugging, and platform incompatibility for small graphs (15–35 nodes).
   - *Resolution:* Drop Apache AGE entirely. Persist graph adjacency in PostgreSQL as JSONB (`nodes` + `edges`). On query or edit, load the graph into an in-memory NetworkX `DiGraph` which executes d-separation, ancestor queries, and cycle checks in sub-5ms.

3. **Re-implementing Bayes-Ball / Backdoor Criterion (§7.3).** Writing a custom 500-line recursive Bayes-Ball engine from scratch introduces severe bug and timeline risk.
   - *Resolution:* Delegate identification directly to DoWhy. Serialize the NetworkX graph to a DOT format string, construct `CausalModel(graph=dot_str)`, call `identify_effect()`, and parse the minimal backdoor adjustment set from the returned estimand.

4. **Attrition as a survival / time-to-event process (§5.3).** Attrition is a duration hazard process, not purely static cross-sectional data.
   - *Resolution:* V1 acknowledges cross-sectional odds-ratio approximation as a non-goal for full continuous-time survival analysis. Domain rules enforce employee tenure (`YearsAtCompany`) as a **mandatory adjustment variable** when estimating effects on `Attrition`.

5. **Arbitrary refutation thresholds vs. statistical testing (§9.3).** Hardcoded thresholds (e.g., `|ATE_placebo| < 0.1 × |ATE_original|`, `CV < 0.3`) violate statistical principles on small-effect or high-variance estimates.
   - *Resolution:* Adopt DoWhy's statistical metrics: Placebo test requires $p > 0.05$ (effect indistinguishable from zero); Random Common Cause requires new point estimate to fall within the original 95% CI; Data Subset requires all 5 bootstrap subsets (80%) to fall inside the original 95% CI with no sign flips.

6. **Propensity trimming changing the estimand (§7.4).** Trimming IPW weights at 1st/99th percentiles changes the target population from the full population to the overlap subpopulation.
   - *Resolution:* If > 5% of weights are trimmed, set `estimand_shifted: true` and surface an explicit warning in the API and UI explaining the result applies only to units with moderate propensity scores.

7. **Multi-valued treatment binarization (§6.1, §10.2).** Real datasets have multi-valued interventions (e.g. `Education`, `JobLevel`). Forcing manual CSV editing hurts UX.
   - *Resolution:* Provide native one-click reference-level binarization in the Ingestion API and UI (e.g., `Education == 'Doctorate' -> Is_Doctorate = 1 | 0`).

8. **Dense 35-node graph "hairball" UX (§10.2).** Connecting 35 nodes on a 2D canvas is tedious and cluttered.
   - *Resolution:* Build an **Adjacency Matrix View** (a 35×35 checkbox grid) as a synchronised tab alongside the React Flow canvas. Checking cells fires the same edge endpoints.

9. **NOTEARS on high-dimensional / small-N data (§8.1).** Optimizing 1,225 parameters on 1,470 rows causes solver divergence and edge flipping.
   - *Resolution:* For datasets with > 20 columns, apply an association pre-filter ($|r| \ge 0.10$ against treatment or outcome) to cap active variables $\le 18$, raise the default sparsity penalty to $\lambda_1 = 0.5$, and display an exploratory heuristic disclaimer.

10. **SSE for task progress (§10.1, §12.3).** Celery task progress must stream cleanly to the frontend.
    - *Resolution:* Use `sse-starlette` with Redis pub/sub. Celery workers publish state transitions (`queued`, `running`, `complete`, `failed`).

---

## 2. Phase 0: Project scaffold + data ingestion

Goal: a working FastAPI application that accepts CSV uploads, validates them, stores as Parquet, and returns dataset profiles. No causal logic yet.

### 2.1 Tickets

| # | Ticket | Acceptance criteria |
|---|---|---|
| 0.1 | **Project scaffold** | `pyproject.toml` with `uv`, ruff, mypy (strict), pytest. Docker Compose (PostgreSQL JSONB + Redis) and native local run mode. `Makefile` with `dev`, `test`, `lint`. CI with GitHub Actions. |
| 0.2 | **Settings and config** | `src/causalforge/settings.py` with Pydantic Settings. Env prefix `CAUSALFORGE_`. Database URL, Redis URL, data directory, log level. |
| 0.3 | **Database models** | SQLAlchemy + Alembic. Tables: `datasets`, `causal_graphs`, `estimation_runs` (standard JSONB for graph nodes/edges). Initial migration. |
| 0.4 | **Dataset ingestion & binarization endpoint** | `POST /api/v1/datasets` accepts CSV/Parquet file + optional column types and treatment binarization rules (`source_column`, `reference_value`, `target_name`). Auto-detects types, validates quality gates, stores Parquet and JSONB metadata. Returns `dataset_id` + profile. |
| 0.5 | **Dataset listing and detail endpoints** | `GET /api/v1/datasets` returns all datasets with metadata. `GET /api/v1/datasets/{id}` returns full profile including column stats and binarized features. |
| 0.6 | **IBM HR and Census data loaders** | CLI commands `causalforge load ibm-hr` and `causalforge load census` that ingest local or remote datasets into the platform. |
| 0.7 | **Pydantic schemas** | Request/response models for all endpoints. `DatasetCreate`, `DatasetResponse`, `DatasetProfile`, `ColumnMetadata`, `TreatmentBinarization`. |
| 0.8 | **Tests** | Unit tests for ingestion pipeline (parse, validate, type-detect, binarize, impute). Integration test: upload IBM HR CSV → get profile → verify column types, stats, and tenure variable. |

### 2.2 Directory structure after Phase 0

```
causalforge/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── Makefile
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   └── causalforge/
│       ├── __init__.py
│       ├── settings.py
│       ├── main.py              # FastAPI app
│       ├── api/
│       │   ├── __init__.py
│       │   ├── router.py
│       │   └── datasets.py      # Dataset endpoints
│       ├── models/
│       │   ├── __init__.py
│       │   └── database.py      # SQLAlchemy models
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── datasets.py      # Pydantic schemas
│       ├── services/
│       │   ├── __init__.py
│       │   └── ingestion.py     # Ingestion pipeline
│       └── cli.py               # CLI commands
├── tests/
│   ├── conftest.py
│   ├── test_ingestion.py
│   └── fixtures/
│       └── sample_hr.csv
├── alembic/
│   ├── alembic.ini
│   └── versions/
└── .github/
    └── workflows/
        └── ci.yml
```

### 2.3 Exit criteria

- `POST /api/v1/datasets` returns 200 with correct profile for IBM HR CSV.
- Column auto-detection produces: Age=continuous, Attrition=binary, Department=categorical, JobSatisfaction=ordinal.
- Quality gates reject a file with all-null columns (422).
- Parquet file on disk matches the uploaded data exactly (round-trip test).
- `docker compose up` starts all services; `make test` passes.

---

## 3. Phase 1: Graph store + DAG editor

Goal: users can create, view, and edit causal DAGs in a React Flow frontend. Graphs are stored in PostgreSQL. Cycle detection works. No estimation yet.

### 3.1 Tickets

| # | Ticket | Acceptance criteria |
|---|---|---|
| 1.1 | **Graph store (NetworkX backend)** | `GraphStore` protocol with `add_node`, `add_edge`, `remove_edge`, `get_parents`, `get_descendants`, `has_cycle`, `to_dot`. NetworkX implementation stores adjacency in PostgreSQL as JSONB. |
| 1.2 | **Graph API endpoints** | `POST /api/v1/graphs` creates empty graph for a dataset (nodes = dataset columns). `GET /api/v1/graphs/{id}` returns nodes + edges. `POST /api/v1/graphs/{id}/edges` adds edge with cycle check (400 on cycle). `DELETE /api/v1/graphs/{id}/edges/{eid}` removes edge. |
| 1.3 | **React project scaffold** | Vite + React 18 + TypeScript. ESLint, Prettier. Proxy to FastAPI in dev. `frontend/` directory. |
| 1.4 | **Dataset registry view (V1)** | Lists datasets. Upload form. Column type editor with multi-valued treatment binarization. Select dataset to proceed to DAG editor. |
| 1.5 | **DAG editor view (V2 Canvas)** | React Flow canvas. Nodes created from dataset columns, coloured by type. Drag to connect = add edge (API call). Right-click edge = remove. Visual cycle rejection (edge turns red, shake animation, reverts). Minimap. Controls. |
| 1.6 | **Adjacency matrix editor view (V2 Matrix)** | Synchronized $N \times N$ checkbox grid tab alongside React Flow canvas. Checking/unchecking cells triggers edge add/remove endpoints. Enables bulk editing without 35-node canvas hairballs. |
| 1.7 | **Tests** | Unit: cycle detection on known graphs. Integration: create graph → add edges → verify via GET → attempt cycle → verify 400. Frontend: Playwright test for edge add/remove in both Canvas and Matrix views. |

### 3.2 Frontend directory structure

```
frontend/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── index.html
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── api/
│   │   └── client.ts           # Axios instance + React Query hooks
│   ├── stores/
│   │   └── graphStore.ts       # Zustand store
│   ├── components/
│   │   ├── DatasetList.tsx
│   │   ├── DatasetUpload.tsx
│   │   ├── ColumnTypeEditor.tsx
│   │   ├── TreatmentBinarizer.tsx
│   │   ├── DAGEditor.tsx       # React Flow wrapper
│   │   ├── AdjacencyMatrix.tsx # Dense checkbox grid view
│   │   ├── CausalNode.tsx      # Custom node component
│   │   └── CausalEdge.tsx      # Custom edge component
│   ├── views/
│   │   ├── DatasetRegistryView.tsx
│   │   ├── DAGEditorView.tsx
│   │   └── Layout.tsx
│   └── styles/
│       ├── index.css
│       ├── variables.css
│       └── components/
└── tests/
    └── dag-editor.spec.ts
```

### 3.3 Exit criteria

- User can upload IBM HR CSV, see 35 nodes in the DAG editor.
- User can toggle edges rapidly in either the React Flow canvas or the Adjacency Matrix view.
- Attempting `OverTime → Age → OverTime` (cycle) is rejected with visual feedback and 400 response.
- Graph persists across page reloads via PostgreSQL JSONB.
- Nodes are coloured by type (continuous=blue, binary=orange, categorical=green, ordinal=purple).

---

## 4. Phase 2: Identification + estimation API

Goal: given a DAG, the system identifies valid adjustment sets and estimates ATE using PSM or IPW. No refutation yet.

### 4.1 Tickets

| # | Ticket | Acceptance criteria |
|---|---|---|
| 2.1 | **DoWhy identification adapter** | Serializes NetworkX graph to DOT, invokes DoWhy `CausalModel.identify_effect()`, and parses minimal backdoor adjustment sets. Replaces custom Bayes-Ball. |
| 2.2 | **Identification API** | `POST /api/v1/identify {graph_id, treatment, outcome}` returns adjustment sets. Frontend highlights adjustment set nodes in both Canvas and Matrix views. |
| 2.3 | **Propensity score model** | Logistic regression on adjustment set. Handles column types (one-hot for categorical, direct for continuous/ordinal/binary). Returns scores + balance diagnostics (SMD before/after). |
| 2.4 | **PSM estimator** | 1:1 nearest-neighbour matching on logit(e). Caliper = 0.2 × σ. Returns ATE + CI + balance diagnostics. Wraps DoWhy. |
| 2.5 | **IPW estimator & estimand shift warning** | Horvitz-Thompson weights. Trimming at 1st/99th percentile. If > 5% weights trimmed, flags `estimand_shifted=True` with overlap population warning. Returns ATE + CI. Wraps DoWhy. |
| 2.6 | **Celery task for estimation** | `POST /api/v1/estimate` enqueues Celery task. Task loads data, runs identification, runs estimation, writes result to DB. `GET /api/v1/estimate/{run_id}` returns result. |
| 2.7 | **Estimation panel (V3)** | Frontend: select treatment + outcome dropdowns. "Identify" button shows adjustment set. "Estimate" button (PSM/IPW selector). Loading state. Result display: ATE ± CI. |
| 2.8 | **IV estimator (optional)** | If `instrument` is provided, use CausalML's IV estimator. Validates instrument topology in DAG. First-stage F-statistic check. |
| 2.9 | **Tests** | Unit: identification on 5 hand-crafted DAGs (chain, fork, collider, M-bias, butterfly) matching ground truth. Integration: IBM HR OverTime→Attrition (conditioning on tenure `YearsAtCompany` + Age) returns valid ATE. |

### 4.2 Identification test cases

| # | DAG | Treatment | Outcome | Expected adjustment set | Why |
|---|---|---|---|---|---|
| T1 | A → T → Y, A → Y | T | Y | {A} | Classic confounding |
| T2 | T → M → Y | T | Y | {} | No confounding; M is a mediator (DO NOT adjust) |
| T3 | T → Y, T → M ← Y | T | Y | {} | M is a collider (DO NOT adjust) |
| T4 | A → T, A → M, M → Y, T → Y | T | Y | {A} | A is the only confounder; M is a mediator |
| T5 | Z → A → T, Z → B → Y, A → Y, T → Y | T | Y | {A} | Z is an ancestor of both but A suffices |

### 4.3 Exit criteria

- All 5 identification test cases pass through DoWhy's `identify_effect()`.
- PSM on IBM HR (OverTime→Attrition, adjusting for Age + YearsAtCompany + JobSatisfaction + MonthlyIncome) returns ATE with CI excluding zero.
- IPW on the same query returns a consistent ATE (within 20% of PSM).
- SMD after matching is < 0.1 for all covariates.
- When > 5% of weights are trimmed, `estimand_shifted` warning flag is returned.

---

## 5. Phase 3: Refutation pipeline

Goal: every estimation is automatically refuted. The system never reports an effect without statistical refutation results.

### 5.1 Tickets

| # | Ticket | Acceptance criteria |
|---|---|---|
| 3.1 | **Placebo treatment refutation** | Replace treatment with random permutation. Re-estimate via DoWhy. Statistical pass criterion: two-sided refutation $p$-value > 0.05. |
| 3.2 | **Random common cause refutation** | Add independent noise variable to adjustment set. Re-estimate via DoWhy. Statistical pass criterion: refutation ATE falls within original 95% CI and difference test $p > 0.05$. |
| 3.3 | **Data subset refutation** | Re-estimate on 5 random 80% bootstrap subsets. Statistical pass criterion: all 5 subset ATEs fall within original 95% CI with zero sign flips. |
| 3.4 | **Robustness judgement** | `is_robust = all three tests pass statistical criteria`. Integrated into estimation Celery task. |
| 3.5 | **Refutation display** | Frontend: statistical pass/fail checklist with exact $p$-values and CI overlaps. Detailed explanation modal. |
| 3.6 | **Non-robust effect UX flow** | Return 200 with `is_robust: false` and full test diagnostics. Frontend displays high-visibility warning banner. |
| 3.7 | **Tests** | Unit: placebo on known causal effect → passes ($p > 0.05$). Placebo on spurious correlation → fails ($p \le 0.05$). Integration: IBM HR OverTime→Attrition → all three refutations pass. RandomColumn→Attrition → placebo fails. |

### 5.2 Exit criteria

- OverTime → Attrition (IBM HR) with correct DAG: all 3 refutations pass statistical criteria ($p > 0.05$, CI overlap), `is_robust = true`.
- RandomColumn → Attrition: placebo test fails ($p \le 0.05$), `is_robust = false`.
- Random common cause test point estimate falls inside original 95% CI.
- Data subset bootstrap produces zero sign reversals.

---

## 6. Phase 4: NOTEARS auto-discovery

Goal: users click "Discover Graph" and the system proposes a causal structure from the raw data with dimensionality safeguards.

### 6.1 Tickets

| # | Ticket | Acceptance criteria |
|---|---|---|
| 4.1 | **NOTEARS solver with sparsity tuning** | Python/SciPy implementation. Linear SEM score. Augmented Lagrangian with L-BFGS-B. Acyclicity constraint via matrix exponential trace. Default $\lambda_1 = 0.5$ for $d > 20$. |
| 4.2 | **Association pre-filtering & preprocessing** | When $d > 20$ (e.g. IBM HR), pre-filter features with low absolute correlation ($|r| < 0.10$) to treatment/outcome, bounding active dimensions $\le 18$. Standardise continuous columns. Encode categoricals. |
| 4.3 | **Thresholding and post-processing** | Threshold |w| < 0.3. Verify result is a DAG (topological sort). If not, remove weakest edges until DAG. |
| 4.4 | **Celery task for discovery** | `POST /api/v1/discover {dataset_id, treatment, outcome}` enqueues task. Worker runs filtered NOTEARS. Writes proposed graph with `source='notears', accepted=false`. Publishes completion via Redis. |
| 4.5 | **Discovery viewer (V5)** | Frontend: side-by-side view with heuristic warning banner. Diff highlighting (green = discovered-only, red = user-only, grey = shared). "Accept" copies discovered graph. "Reject" discards. |
| 4.6 | **SSE for task progress** | `GET /api/v1/events?task_id=...` returns SSE stream. Worker publishes: `queued`, `running`, `complete`, `failed`. Frontend shows progress indicator. |
| 4.7 | **Tests** | Unit: NOTEARS on 5-variable linear SEM with known DAG → SHD ≤ 2. Integration: discover on pre-filtered IBM HR converges in < 25s with no cycles. |

### 6.2 Synthetic test case for NOTEARS

```python
# Known DAG: X1 → X3, X2 → X3, X3 → X4, X1 → X4
# SEM: X1 ~ N(0,1), X2 ~ N(0,1), X3 = 0.8*X1 + 0.5*X2 + ε, X4 = 0.6*X3 + 0.3*X1 + ε
# Expected: NOTEARS recovers all 4 edges with correct orientation
# SHD tolerance: ≤ 2 (may miss one edge or reverse one)
```

### 6.3 Exit criteria

- NOTEARS on the synthetic test case: SHD ≤ 2.
- NOTEARS on IBM HR (35 columns): completes in < 60s, returns a valid DAG, has ≥ 15 edges.
- Discovery viewer shows the proposed graph with diff highlighting.
- "Accept" creates a new graph version that can be used for estimation.
- SSE stream delivers progress updates to the frontend.

---

## 7. Phase 5: Results dashboard + polish

Goal: a polished results view with forest plots, balance diagnostics, and comparison features. UI polish.

### 7.1 Tickets

| # | Ticket | Acceptance criteria |
|---|---|---|
| 5.1 | **Results dashboard (V4)** | Forest plot for ATE + CI (Recharts). Refutation checklist. Balance diagnostics (SMD lollipop chart before/after). |
| 5.2 | **Estimation history** | List of all estimation runs for a dataset. Click to view details. Filter by treatment, outcome, method. |
| 5.3 | **Multi-method comparison** | Run PSM and IPW on the same query. Side-by-side ATE comparison in the dashboard. |
| 5.4 | **Export** | Download estimation results as JSON or CSV. Download DAG as DOT or PNG. |
| 5.5 | **UI polish** | Dark mode support. Loading skeletons. Error boundaries. Responsive layout. Tooltips for causal concepts (confounder, collider, d-separation). |
| 5.6 | **Educational tooltips** | Hover over "Backdoor Criterion" → explanation. Hover over "Collider" warning → explanation with diagram. Hover over refutation test name → what it tests and why. |
| 5.7 | **Tests** | Playwright E2E: full flow from upload → draw DAG → identify → estimate → view results → export. |

### 7.2 Exit criteria

- Full E2E test passes: upload IBM HR → draw DAG → estimate OverTime→Attrition → see ATE forest plot → see refutation checklist → export JSON.
- PSM and IPW comparison shows consistent ATEs.
- Dark mode toggle works.
- All tooltips render correctly.

---

## 8. Phase 6: Testing, validation, documentation

Goal: comprehensive test suite, semi-synthetic benchmarks, documentation, and deployment guide.

### 8.1 Tickets

| # | Ticket | Acceptance criteria |
|---|---|---|
| 6.1 | **Semi-synthetic benchmark suite** | `scripts/generate_semi_synthetic.py` creates 3 benchmarks: (a) single confounder, (b) multiple confounders, (c) collider bias trap. Each has a ground-truth ATE. |
| 6.2 | **Benchmark evaluation** | `scripts/run_benchmarks.py` runs identification + estimation on all 3 benchmarks. Reports ATE bias, CI coverage, refutation pass rates. CI-blocking. |
| 6.3 | **Census Income integration** | End-to-end test with Census data: upload → auto-detect types → discover → estimate Education→Income. |
| 6.4 | **API documentation** | OpenAPI spec auto-generated by FastAPI. Additional narrative docs in `docs/`. |
| 6.5 | **Deployment guide** | `docs/DEPLOYMENT.md`: Docker Compose for local, VPS deployment with nginx + HTTPS. Environment variables reference. |
| 6.6 | **README** | Project overview, quickstart, architecture diagram, links to docs. |
| 6.7 | **Performance benchmarks** | Estimation latency on IBM HR: < 10s (PSM), < 5s (IPW). Discovery latency on IBM HR: < 60s. API response time: < 200ms for sync endpoints. |
| 6.8 | **Security review** | File upload validation (size limits, extension checks). SQL injection prevention (parameterised queries). CORS configuration. |

### 8.2 Semi-synthetic benchmark details

| Benchmark | True DAG | Treatment | Outcome | True ATE | Confounders | Trap |
|---|---|---|---|---|---|---|
| B1: Single confounder | Age → {T, Y}, T → Y | OverTime (binary) | Attrition (binary) | 0.15 | Age | Naive estimate is biased upward |
| B2: Multiple confounders | Age → {T, Y}, JobSat → {T, Y}, Income → Y, T → Y | OverTime | Attrition | 0.12 | Age, JobSatisfaction | Must adjust for both |
| B3: Collider trap | T → Y, T → M, Y → M | OverTime | Attrition | 0.15 | None | Adjusting for M (collider) introduces bias |

### 8.3 Exit criteria

- All 3 benchmarks: |ATE_estimated - ATE_true| < 0.05 (absolute bias).
- CI coverage ≥ 88% (out of 100 random seeds) on all benchmarks.
- B3: system correctly identifies {} as the adjustment set (does NOT include M).
- Census end-to-end: runs without error; discovered graph has ≥ 8 edges.
- All CI checks pass. README renders correctly on GitHub.

---

## 9. Dependency inventory

### 9.1 Backend (Python 3.12)

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | ≥ 0.110 | API framework |
| `uvicorn[standard]` | ≥ 0.29 | ASGI server |
| `sqlalchemy` | ≥ 2.0 | ORM |
| `alembic` | ≥ 1.13 | Migrations |
| `asyncpg` | ≥ 0.29 | Async PostgreSQL driver |
| `pydantic` | ≥ 2.7 | Validation |
| `pydantic-settings` | ≥ 2.3 | Configuration |
| `celery[redis]` | ≥ 5.4 | Task queue |
| `redis` | ≥ 5.0 | Cache + pub/sub |
| `sse-starlette` | ≥ 2.0 | Server-sent events |
| `dowhy` | ≥ 0.11, < 0.12 | Causal inference |
| `scikit-learn` | ≥ 1.5 | Propensity models |
| `pandas` | ≥ 2.2 | Data manipulation |
| `pyarrow` | ≥ 16.0 | Parquet I/O |
| `networkx` | ≥ 3.3 | Graph operations (fallback) |
| `scipy` | ≥ 1.13 | NOTEARS optimisation |
| `numpy` | ≥ 1.26 | Numerics |
| `econml` | ≥ 0.15 (optional) | CausalML IV estimators |

### 9.2 Frontend (Node 20)

| Package | Version | Purpose |
|---|---|---|
| `react` | ≥ 18.3 | UI framework |
| `react-dom` | ≥ 18.3 | DOM rendering |
| `@xyflow/react` | ≥ 12.0 | DAG editor (React Flow) |
| `zustand` | ≥ 4.5 | State management |
| `axios` | ≥ 1.7 | HTTP client |
| `@tanstack/react-query` | ≥ 5.50 | Data fetching |
| `recharts` | ≥ 2.12 | Charts |
| `typescript` | ≥ 5.5 | Type safety |
| `vite` | ≥ 5.3 | Build tool |

---

## 10. Open implementation questions

| # | Question | Decision needed by | Current leaning |
|---|---|---|---|
| IQ1 | NetworkX fallback: keep permanently or drop after AGE is stable? | Phase 1 end | Keep for testing; AGE for production |
| IQ2 | Should the CLI (`causalforge` command) be a Typer or Click app? | Phase 0 start | Typer (Pydantic integration) |
| IQ3 | Should we use Alembic autogenerate or write migrations manually? | Phase 0 start | Autogenerate + review |
| IQ4 | Frontend testing: Playwright only or also Vitest for unit tests? | Phase 1 start | Both: Vitest for components, Playwright for E2E |
| IQ5 | Should propensity model be configurable (logistic vs. GBM) from the API? | Phase 2 | Yes, with logistic as default |
| IQ6 | Should discovery support user-specified edge constraints (forced/forbidden edges)? | Phase 4 | Defer to v2; complexity is high |
