# Managed agent spike

Branch: `feature/managed-agent-spike`. Experimental local CLI, no daemon API
or UI yet. Uses the existing Claux engine, configured model profiles, session
SQLite store, and filesystem containment. No llm-mux source changes needed.

Build with `cargo build`. Use `target/debug/claux` explicitly; this does not
replace an installed binary or the release binary used by frozen benchmarks.

```bash
./target/debug/claux task start 'Inspect the failing tests and propose a fix' \
  --cwd /absolute/path/to/repo --model my-profile --worktree
./target/debug/claux task list
./target/debug/claux task status TASK_ID
./target/debug/claux task attach TASK_ID --follow
./target/debug/claux task wait TASK_ID
./target/debug/claux task send TASK_ID 'Now add a regression test'
./target/debug/claux task cancel TASK_ID
./target/debug/claux task retry TASK_ID   # only a failed launch, before a turn began
```

These commands are also safe to enter directly in Nushell. Substitute the exact
32-character ID printed by `start`; prefixes are not accepted. Commands print
JSON; `attach` prints newline-delimited versioned JSON events. Closing an attach
client does not cancel its worker. `wait` prints a versioned one-shot result and
exits nonzero for failed/cancelled turns. Its own timeout leaves the worker alone.

At most four managed workers run per user data directory; excess launches
fail before evaluation and can be retried after a slot opens.
Each turn starts a detached process, not a permanent daemon. Follow-ups between
turns reopen the saved conversation and archive. Active-turn steering and
queued follow-ups are deliberately absent. OS file locks enforce one worker
and serialize control operations without trusting reusable PIDs. The initial
launch has a ten-second grace period before a missing worker is interrupted.
The `--timeout` on start bounds each model/tool turn (default 900, max 3600 seconds).
Provider/MCP initialization uses existing Claux bounds and precedes that timer.
Cancellation uses the engine's existing token and preserves repaired history.

`--worktree` creates a `claux-task/ID` branch and worktree from HEAD, excluding
uncommitted changes in the source checkout. It never commits, merges, pushes,
or deletes worktrees. Without it, tasks share the selected working directory;
use worktrees when tasks may edit concurrently. Review diffs with:

```bash
git -C /task/cwd diff
```

Default permissions remain `default`, independent of unattended execution.
Background permission prompts are denied. Opt into `--permission-mode accept-edits`
for edits, or configure specific command permissions. `bypass` is explicit and
still does not disable containment. Project trust is evaluated in the task's
actual directory; pass `--trust-project` only when intended. Global transport
and filesystem overrides are forwarded. Task bindings contain no credentials;
workers resolve credentials from current configuration/environment, and reject
changed provider/model bindings.

Tasks live below the platform's local data directory in `claux/tasks/ID`.
On Unix directories are private (0700), artifacts 0600. Text events and engine
checkpoints can contain private project content. Tool events record names and
status, not raw arguments/results. The event log has a 32 MiB cap; reaching it
cancels the turn. Usage is per turn, not a cumulative task bill.

Crashes are reported as interrupted using lock liveness. Completed and cancelled
turns can receive follow-ups. An interrupted turn cannot be automatically sent
again: inspect `checkpoint.json` and the session archive first, because tools
may already have changed files. A failed launch with its original pending prompt intact can use `task retry`;
this never retries a turn whose prompt was already consumed.
Automatic mid-turn crash recovery, remote access, runtime
approval, scheduling, and task cleanup are follow-up design work. Do not open the
same saved conversation concurrently through the ordinary TUI/REPL.

## llm-mux experiment

Copy `examples/managed-pair.toml` to a trusted `.llm-mux/workflows/` directory
and use an explicit path to this spike binary:

```bash
llm-mux --allow-project-workflows run managed-pair \
  claux=/absolute/path/to/claux/target/debug/claux \
  repo=/absolute/path/to/project model=my-profile \
  prompt='Inspect the code and implement a small improvement with tests'
```

The workflow launches both workers before waiting. Each has a separate branch
and worktree. llm-mux retains the task IDs/results in its existing ledger;
Claux owns conversation history. Cancelling the workflow does not yet cancel
launched tasks: inspect `claux task list` and cancel each explicitly. Resuming
llm-mux after an interrupted launch can create another task if the launch output
was not recorded. This example is for manual experimentation, not unattended
production orchestration.
