# NIOS Terraform Test Agent

An agent that writes test scenarios for an Infoblox NIOS Terraform resource, verifies each one with Terraform, repairs the ones that fail, and learns rules it can reuse next time. It runs locally on Ollama and needs only Python 3.10+ and `requests`.

The agent core (`agent_core/`) knows nothing about Terraform or NIOS. Everything domain-specific lives in `nios/`. You can reuse the core for a completely different agent by writing a new set of tools.

---

## How it was developed, in order

The order matters more than any single piece. Most failed agents start by writing a prompt and connecting a model. This one connected the model last.

**1. Define "done" in code before writing anything else.** The task is "N scenarios that pass `terraform plan` and M negative scenarios that Terraform rejects." That sentence became `make_done_check()` in `nios/toolset.py`. When the model says it's finished, the loop asks this function, not the model. If the check fails, the model is told exactly what's still missing and has to keep going. A model's claim that it succeeded is never evidence.

**2. Find an automatic verifier.** Agents work well when their output can be checked by something other than themselves. `terraform plan` is a free, authoritative judge: it either accepts a file or explains precisely why not. The whole design is built around that feedback loop.

**3. Take ground truth from the source, not from the model.** Field names, types, and which fields are required come from `terraform providers schema -json`, the provider's own definition. The model is never asked to guess or extract a schema.

**4. Make the model's job as small as possible.** Code generates `main.tf` and `variables.tf` from the schema, so the model never writes Terraform. It only supplies values, as structured tool arguments. Scenarios are saved as `.tfvars.json`, so HCL syntax errors are impossible. The resource under test is bound in code, so the model can't test the wrong one or mistype its name. Every decision taken away from the model is one it can't get wrong.

**5. Build and test the tools without the model.** Each tool validates its inputs, returns short, useful output, and explains its errors in a way that says how to fix them. `python cli.py describe` and `python cli.py plan` let you exercise the tools by hand. Tool quality matters more than prompt quality: the model sees the world only through tool descriptions and results.

**6. Keep the loop small, generic, and impossible to crash.** `agent_core/loop.py` is about 160 lines. Unknown tools, malformed arguments, denied actions, and tool exceptions all become messages the model can read and react to. Old tool outputs are compacted so the context window doesn't silently overflow. A hard step limit stops runaway loops.

**7. Put guardrails in code, not in the prompt.** Tools have risk levels. Reading and local writes run freely; anything that touches NIOS needs your approval. File access is locked to the workspace. Secrets reach Terraform only through `TF_VAR_*` environment variables and are redacted from everything sent to the model or written to logs. Applying refuses to run on leftover state, and destroy always runs after apply, even when apply fails.

**8. Record everything.** Every model turn and tool call is written to `workspace/traces/<run>.jsonl`. When a run goes wrong, the trace shows exactly what the model saw and what it decided.

**9. Test the whole agent offline.** `tests/` uses a scripted fake model and a fake Terraform binary to run complete sessions: a scenario that fails, gets repaired, and passes; a model that claims success too early; destructive actions with and without approval. 32 tests run in well under a second, with no Ollama, Terraform, or NIOS needed.

**10. Only then connect a real model, and measure.** `evals/run_evals.py` runs the agent across many resources and compares the solve rate with the previous run. After that, development is a loop: run the evals, read the traces of failures, improve a tool or the prompt, run the evals again.

---

## Architecture

```
cli.py / evals/run_evals.py        entry points
        │
nios/app.py                        wires everything together (one place)
        │
agent_core/                        generic, reusable, no domain knowledge
  loop.py      the agent loop, done checks, context compaction
  llm.py       model interface + Ollama adapter (swap providers here)
  tools.py     tool definitions, argument validation, risk levels
  policy.py    approvals, path sandbox, secret redaction
  memory.py    SQLite: learned notes + verified examples
  tracing.py   JSONL traces
        │
nios/                              domain layer
  toolset.py   the agent's tools, scenario tracking, done check
  terraform.py schema parsing, file generation, plan/apply/destroy
  docs.py      document search (RAG) as a tool
  prompts.py   system prompt and task text
```

The agent's tools for one run are `describe_resource`, `search_docs`, `get_examples`, `write_scenario`, `plan_scenario`, `remember`, and, only when you pass `--apply`, `apply_scenario`.

Memory has two kinds of entries. Notes are rules the agent learned from errors, and it writes those itself. Verified examples are scenario files that Terraform accepted, and only code writes those, at the moment a plan or apply succeeds. The model can't mark its own work as correct.

---

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull nomic-embed-text          # only needed for document search

cp .env.example .env                  # fill in your TEST grid details
set -a; source .env; set +a
```

Before the first real run, check two assumptions against your provider version. The provider block in `render_provider_tf()` (`nios/terraform.py`) uses the arguments `nios_host_url`, `nios_username`, and `nios_password`; adjust them if your version names them differently. The resource names in `evals/cases.json` follow the provider's naming pattern but weren't checked against a live schema; `python cli.py list` shows the real ones.

## Usage

```bash
python -m unittest discover -s tests -t .           # run the test suite

python cli.py list dns                               # resource types containing "dns"
python cli.py describe nios_dns_record_a             # fields, straight from the schema

python cli.py run nios_dns_record_a                  # 3 passing + 1 negative, plan only
python cli.py run nios_dns_record_a --positive 5 --negative 2
python cli.py run nios_dns_record_a --apply          # also create + destroy on NIOS (asks first)

pdftotext -layout wapi_reference.pdf docs/wapi.txt   # optional: enable document search
python cli.py index-docs docs/wapi.txt

python evals/run_evals.py                            # measure across all cases
```

Each run prints a results table and the paths to its trace and `results.json`. Generated files live in `workspace/terraform/<resource_type>/`.

---

## What's verified, and what isn't

The test suite passes, covering the loop, tools, policy, memory, Terraform file generation, error parsing, and complete end-to-end sessions with fakes. It has not been run against a real Ollama server, Terraform binary, or NIOS grid, so treat the first real run as a test of the environment, and read its trace.

Known limits: nested blocks (`block_types` in the schema) aren't supported yet; the agent is told to skip them. Complex fields such as maps and lists are passed through with type `any` and validated by the provider. An 8B model will sometimes pick the wrong tool or give up early. The done check and nudges handle much of that, and the evals tell you whether a larger model is worth it.

## Good next steps

Expose the toolset as an MCP server so other clients can use the same tools. Add support for nested blocks using `dynamic` blocks in the generated `main.tf`. Run the unit tests in CI on every change. Try two or three models against the eval set and let the numbers decide.
