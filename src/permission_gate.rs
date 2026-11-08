//! Fail-closed, per-action authorization for supervised one-shot runs.
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{path::Path, process::Stdio, time::Duration};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const TIMEOUT: Duration = Duration::from_secs(10);
const LIMIT: u64 = 16 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Response {
    request_id: String,
    decision: Decision,
}
#[derive(Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Decision {
    Allow,
    Deny,
}

pub async fn authorize(
    path: &Path,
    tool: &str,
    input: &Value,
    read_only: bool,
    proposed: &str,
) -> Result<bool> {
    authorize_with_timeout(path, tool, input, read_only, proposed, TIMEOUT).await
}

async fn authorize_with_timeout(
    path: &Path,
    tool: &str,
    input: &Value,
    read_only: bool,
    proposed: &str,
    timeout: Duration,
) -> Result<bool> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let cwd = std::env::current_dir()?.canonicalize()?;
    let request = serde_json::to_vec(
        &json!({"schema_version":1,"request_id":request_id,"tool":tool,"input":input,"cwd":cwd,"read_only":read_only,"proposed_decision":proposed}),
    )?;
    ensure!(
        request.len() <= 1024 * 1024,
        "permission request exceeds limit"
    );
    let mut child = tokio::process::Command::new(path)
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .context("permission gate could not start")?;
    let mut stdin = child
        .stdin
        .take()
        .context("permission gate stdin unavailable")?;
    let stdout = child
        .stdout
        .take()
        .context("permission gate stdout unavailable")?;
    let work = async {
        stdin.write_all(&request).await?;
        stdin.write_all(b"\n").await?;
        stdin.shutdown().await?;
        drop(stdin);
        let mut output = Vec::new();
        stdout.take(LIMIT + 1).read_to_end(&mut output).await?;
        ensure!(
            output.len() as u64 <= LIMIT,
            "permission response exceeds limit"
        );
        let status = child.wait().await?;
        ensure!(status.success(), "permission gate failed");
        let response: Response =
            serde_json::from_slice(&output).context("invalid permission response")?;
        ensure!(
            response.request_id == request_id,
            "permission response does not match request"
        );
        Ok::<bool, anyhow::Error>(response.decision == Decision::Allow)
    };
    tokio::time::timeout(timeout, work)
        .await
        .context("permission gate timed out")?
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn script(root: &Path, body: &str) -> std::path::PathBuf {
        let path = root.join("gate");
        std::fs::write(
            &path,
            format!("#!/usr/bin/python3\nimport json,sys,time\n{body}\n"),
        )
        .unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        path
    }

    #[tokio::test]
    async fn response_is_bound_to_unique_request_and_exact_input() {
        let root = tempfile::tempdir().unwrap();
        let path=script(root.path(),"r=json.load(sys.stdin); assert r['tool']=='Write' and r['input']=={'path':'a'}; print(json.dumps({'request_id':r['request_id'],'decision':'allow'}))");
        assert!(
            authorize(&path, "Write", &json!({"path":"a"}), false, "ask")
                .await
                .unwrap()
        );
        let path = script(
            root.path(),
            "r=json.load(sys.stdin); print(json.dumps({'request_id':'stale','decision':'allow'}))",
        );
        assert!(
            authorize(&path, "Write", &json!({"path":"a"}), false, "ask")
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn malformed_failed_empty_and_timed_out_gates_fail_closed() {
        let root = tempfile::tempdir().unwrap();
        for body in ["sys.exit(1)","print('not-json')","pass","r=json.load(sys.stdin); print(json.dumps({'request_id':r['request_id'],'decision':'allow','extra':1}))"] {
            let path=script(root.path(),body);
            assert!(authorize(&path,"Write",&json!({}),false,"ask").await.is_err());
        }
        let path = script(root.path(), "time.sleep(5)");
        assert!(authorize_with_timeout(
            &path,
            "Write",
            &json!({}),
            false,
            "ask",
            Duration::from_millis(50)
        )
        .await
        .is_err());
        let path=script(root.path(),"r=json.load(sys.stdin); print(json.dumps({'request_id':r['request_id'],'decision':'deny'}))");
        assert!(!authorize(&path, "Write", &json!({}), false, "ask")
            .await
            .unwrap());
    }
}
