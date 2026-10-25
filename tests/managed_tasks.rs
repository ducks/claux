//! Exercise detached workers against a local provider, with isolated storage.
#![cfg(target_os = "linux")]
use serde_json::{json, Value};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Output};
use std::sync::mpsc;
use std::time::Duration;

fn cli(home: &Path, args: &[&str]) -> Output {
    let output = Command::new(env!("CARGO_BIN_EXE_claux"))
        .env("XDG_DATA_HOME", home.join("data"))
        .env("XDG_CONFIG_HOME", home.join("config"))
        .env("HOME", home)
        .env("CLAUX_TEST_KEY", "local-fixture-key")
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{:?}: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn json_output(output: Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn detached_followups_worktrees_and_event_replay() {
    let home = tempfile::tempdir().unwrap();
    let repo = home.path().join("repo");
    fs::create_dir(&repo).unwrap();
    for args in [
        vec!["init"],
        vec![
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "initial",
        ],
    ] {
        assert!(Command::new("git")
            .current_dir(&repo)
            .args(args)
            .output()
            .unwrap()
            .status
            .success());
    }
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    let (requests_tx, requests_rx) = mpsc::channel();
    let (stop_tx, stop_rx) = mpsc::channel();
    let server = std::thread::spawn(move || {
        let mut count = 0;
        loop {
            if stop_rx.try_recv().is_ok() {
                break;
            }
            let (mut stream, _) = match listener.accept() {
                Ok(s) => s,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }
                Err(e) => panic!("{e}"),
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(value) = line.to_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let request: Value = serde_json::from_slice(&body).unwrap();
            requests_tx.send(request).unwrap();
            count += 1;
            if count == 3 {
                // Leave this stream open long enough to exercise cancellation.
                std::thread::sleep(Duration::from_secs(2));
                continue;
            }
            let text = if count == 1 {
                "first answer"
            } else {
                "follow-up answer"
            };
            let payload = format!(
                "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                json!({"id":"fixture","choices":[{"index":0,"delta":{"role":"assistant","content":text},"finish_reason":null}]}),
                json!({"id":"fixture","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2}})
            );
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", payload.len(), payload).unwrap();
        }
    });
    let config = home.path().join("test.toml");
    fs::write(
        &config,
        format!(
            r#"
default_model = "test"
[providers.fixture]
type = "openai"
base_url = "http://127.0.0.1:{port}/v1"
api_key_env = "CLAUX_TEST_KEY"
[model_profiles.test]
provider = "fixture"
model = "fixture-model"
context_window = 32000
"#
        ),
    )
    .unwrap();
    let started = json_output(cli(
        home.path(),
        &[
            "--config",
            config.to_str().unwrap(),
            "task",
            "start",
            "remember this",
            "--cwd",
            repo.to_str().unwrap(),
            "--worktree",
            "--model",
            "test",
            "--timeout",
            "20",
        ],
    ));
    let id = started["id"].as_str().unwrap();
    assert!(started["cwd"].as_str().unwrap().ends_with("worktree"));
    assert!(started["branch"]
        .as_str()
        .unwrap()
        .starts_with("claux-task/"));
    let first = json_output(cli(home.path(), &["task", "wait", id, "--timeout", "30"]));
    assert_eq!(first["result"], "first answer");
    cli(home.path(), &["task", "send", id, "continue that thought"]);
    let second = json_output(cli(home.path(), &["task", "wait", id, "--timeout", "30"]));
    assert_eq!(second["result"], "follow-up answer");
    let first_request = requests_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let second_request = requests_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(
        second_request["messages"].as_array().unwrap().len()
            > first_request["messages"].as_array().unwrap().len()
    );
    assert!(second_request.to_string().contains("first answer"));
    let events = cli(home.path(), &["task", "attach", id]);
    assert!(String::from_utf8(events.stdout)
        .unwrap()
        .contains("turn_completed"));
    let state = json_output(cli(home.path(), &["task", "status", id]));
    assert_eq!(state["status"], "idle");
    assert_eq!(state["worker_active"], false);
    let artifact_dir = Path::new(started["artifacts"].as_str().unwrap());
    assert!(!fs::read_to_string(artifact_dir.join("task.json"))
        .unwrap()
        .contains("local-fixture-key"));
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(
        fs::metadata(artifact_dir).unwrap().permissions().mode() & 0o777,
        0o700
    );
    assert_eq!(
        fs::metadata(artifact_dir.join("events.jsonl"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    cli(home.path(), &["task", "send", id, "wait for cancellation"]);
    requests_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    cli(home.path(), &["task", "cancel", id]);
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let state = json_output(cli(home.path(), &["task", "status", id]));
        if state["status"] == "cancelled" && state["worker_active"] == false {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "cancel did not stop the worker"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    // A changed transport must fail closed rather than silently change providers.
    let changed = fs::read_to_string(&config)
        .unwrap()
        .replace("/v1", "/different");
    fs::write(&config, changed).unwrap();
    cli(
        home.path(),
        &["task", "send", id, "should not reach a provider"],
    );
    let failed = Command::new(env!("CARGO_BIN_EXE_claux"))
        .env("XDG_DATA_HOME", home.path().join("data"))
        .args(["task", "wait", id, "--timeout", "10"])
        .output()
        .unwrap();
    assert!(!failed.status.success());
    assert!(requests_rx.try_recv().is_err());
    // Restore the pinned transport, then retry only the unconsumed launch.
    fs::write(
        &config,
        fs::read_to_string(&config)
            .unwrap()
            .replace("/different", "/v1"),
    )
    .unwrap();
    cli(home.path(), &["task", "retry", id]);
    let retried = json_output(cli(home.path(), &["task", "wait", id, "--timeout", "30"]));
    assert_eq!(retried["result"], "follow-up answer");
    requests_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    stop_tx.send(()).unwrap();
    server.join().unwrap();
}
