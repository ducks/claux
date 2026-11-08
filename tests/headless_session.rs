#![cfg(target_os = "linux")]
use serde_json::{json, Value};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    path::Path,
    process::Command,
    sync::mpsc,
};

fn run(home: &Path, config: &Path, args: &[&str]) -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_claux"))
        .current_dir(home)
        .env("HOME", home)
        .env("XDG_DATA_HOME", home.join("data"))
        .env("XDG_CONFIG_HOME", home.join("config"))
        .env("CLAUX_TEST_KEY", "fixture")
        .arg("--config")
        .arg(config)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
#[test]
fn one_shot_session_persists_and_resumes_with_prior_history() {
    let home = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        for n in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut len = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; len];
            reader.read_exact(&mut body).unwrap();
            tx.send(serde_json::from_slice::<Value>(&body).unwrap())
                .unwrap();
            let chunk = json!({"choices":[{"index":0,"delta":{"content":format!("answer-{n}")},"finish_reason":"stop"}]});
            let payload = format!("data: {chunk}\n\ndata: [DONE]\n\n");
            write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",payload.len(),payload).unwrap();
        }
    });
    let config = home.path().join("config.toml");
    fs::write(&config,format!("default_model = \"test\"\n[providers.fixture]\ntype = \"openai\"\nbase_url = \"http://127.0.0.1:{port}/v1\"\napi_key_env = \"CLAUX_TEST_KEY\"\n[model_profiles.test]\nprovider = \"fixture\"\nmodel = \"fixture-model\"\ncontext_window = 32000\n")).unwrap();
    let events = home.path().join("events.jsonl");
    let first = run(
        home.path(),
        &config,
        &[
            "--print",
            "first prompt",
            "--persist-session",
            "--output-format",
            "json",
            "--event-log",
            events.to_str().unwrap(),
        ],
    );
    let id = first["session_id"].as_str().unwrap();
    assert_eq!(first["result"], "answer-0");
    let event_text = fs::read_to_string(&events).unwrap();
    let records: Vec<Value> = event_text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records.first().unwrap()["event"], "turn_started");
    assert_eq!(records.last().unwrap()["event"], "turn_finished");
    assert!(records
        .iter()
        .any(|event| event["event"] == "model_request"));
    assert!(!event_text.contains("first prompt"));
    assert!(!event_text.contains("answer-0"));
    let second = run(
        home.path(),
        &config,
        &[
            "--print",
            "second prompt",
            "--resume",
            id,
            "--output-format",
            "json",
        ],
    );
    assert_eq!(second["session_id"], id);
    assert_eq!(second["result"], "answer-1");
    let _first = rx.recv().unwrap();
    let request = rx.recv().unwrap();
    let messages = request["messages"].to_string();
    assert!(messages.contains("first prompt"));
    assert!(messages.contains("answer-0"));
    assert!(messages.contains("second prompt"));
    worker.join().unwrap();
}
