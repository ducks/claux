//! Voice mode for the TUI: push-to-talk input through a transcription
//! command, spoken replies through per-language speech commands.
//!
//! Everything runs as a subprocess named in the global `[voice]` config, the
//! same way `api_key_cmd` names a credential helper: claux links no audio or
//! model libraries and any engine with a command line works. Project config
//! cannot set these commands, because "the command claux runs on every reply"
//! is arbitrary code execution.
//!
//! Replies may mix languages. The model marks target-language spans with
//! `[xx]...[/xx]` tags (ISO code) and `segments` routes each span to the
//! speech command configured for that language; untagged text uses
//! `speak_default`. Tags are left in the transcript as written.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::{Child, Command};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VoiceConfig {
    /// Start the TUI with voice mode on. `/voice` toggles it per session.
    pub enabled: bool,
    /// Records microphone audio to `{file}` until it receives SIGINT.
    pub record_cmd: Option<String>,
    /// Transcribes `{file}` and prints the text to stdout.
    pub transcribe_cmd: Option<String>,
    /// Submit the transcript as soon as it lands instead of leaving it in
    /// the input line for review.
    pub auto_submit: bool,
    /// Speech commands by language code. Each reads text on stdin and
    /// plays it.
    pub speak: BTreeMap<String, String>,
    /// Language used for untagged reply text.
    pub speak_default: Option<String>,
    /// Upper bound on one transcription run.
    pub transcribe_timeout_secs: u64,
}

impl VoiceConfig {
    pub fn transcribe_timeout(&self) -> Duration {
        Duration::from_secs(if self.transcribe_timeout_secs == 0 {
            60
        } else {
            self.transcribe_timeout_secs
        })
    }

    pub fn can_listen(&self) -> bool {
        self.record_cmd.is_some() && self.transcribe_cmd.is_some()
    }

    pub fn can_speak(&self) -> bool {
        !self.speak.is_empty()
    }

    /// Speech command for a language code, falling back to the default
    /// language. None when nothing applies.
    pub fn speak_command(&self, lang: Option<&str>) -> Option<&str> {
        lang.and_then(|code| self.speak.get(code))
            .or_else(|| {
                self.speak_default
                    .as_deref()
                    .and_then(|code| self.speak.get(code))
            })
            .map(String::as_str)
    }

    /// A one-line summary for the status line and `/voice`.
    pub fn describe(&self) -> String {
        let listen = if self.can_listen() {
            "listen"
        } else {
            "no record/transcribe command"
        };
        let speak = if self.can_speak() {
            format!(
                "speak {}",
                self.speak.keys().cloned().collect::<Vec<_>>().join(",")
            )
        } else {
            "no speak command".to_string()
        };
        format!("{listen}; {speak}")
    }
}

/// One span of reply text and the language code it was tagged with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub lang: Option<String>,
    pub text: String,
}

/// Split reply text into language-tagged spans. Tags are `[xx]` and `[/xx]`
/// with a two- or three-letter lowercase code. Anything else in brackets is
/// ordinary text. An unclosed tag runs to the end.
pub fn segments(text: &str) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    let mut lang: Option<String> = None;
    let mut buf = String::new();
    let mut rest = text;

    fn tag_at(s: &str) -> Option<(Option<&str>, usize)> {
        let body = s.strip_prefix('[')?;
        let end = body.find(']')?;
        let inner = &body[..end];
        let (closing, code) = match inner.strip_prefix('/') {
            Some(code) => (true, code),
            None => (false, inner),
        };
        let valid = (2..=3).contains(&code.len()) && code.chars().all(|c| c.is_ascii_lowercase());
        if !valid {
            return None;
        }
        Some((if closing { None } else { Some(code) }, end + 2))
    }

    while !rest.is_empty() {
        if let Some((tag, len)) = tag_at(rest) {
            if !buf.trim().is_empty() {
                out.push(Segment {
                    lang: lang.clone(),
                    text: buf.trim().to_string(),
                });
            }
            buf.clear();
            lang = tag.map(str::to_string);
            rest = &rest[len..];
            continue;
        }
        let ch = rest.chars().next().unwrap();
        buf.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    if !buf.trim().is_empty() {
        out.push(Segment {
            lang,
            text: buf.trim().to_string(),
        });
    }
    out
}

/// Substitute `{file}` with a shell-quoted path.
fn fill_file(template: &str, file: &Path) -> String {
    let quoted = format!("'{}'", file.display().to_string().replace('\'', "'\\''"));
    template.replace("{file}", &quoted)
}

fn shell(command: &str) -> Command {
    #[cfg(windows)]
    let mut process = {
        let mut process = Command::new("cmd");
        process.args(["/D", "/C", command]);
        process
    };
    #[cfg(not(windows))]
    let mut process = {
        let mut process = Command::new("sh");
        process.args(["-c", command]);
        process
    };
    crate::command_sandbox::sanitize_command_environment(&mut process);
    process.kill_on_drop(true);
    process
}

/// A microphone capture in progress.
pub struct Recording {
    child: Child,
    dir: tempfile::TempDir,
    file: PathBuf,
}

impl Recording {
    pub fn start(config: &VoiceConfig) -> Result<Self> {
        let template = config
            .record_cmd
            .as_deref()
            .context("voice.record_cmd is not configured")?;
        let dir = tempfile::Builder::new().prefix("claux-voice-").tempdir()?;
        let file = dir.path().join("input.wav");
        let child = shell(&fill_file(template, &file))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("could not start record_cmd {template:?}"))?;
        Ok(Self { child, dir, file })
    }

    /// Ask the recorder to finish (SIGINT, the convention for audio
    /// capture tools), wait for it, then transcribe the file.
    pub async fn stop_and_transcribe(mut self, config: &VoiceConfig) -> Result<String> {
        #[cfg(unix)]
        if let Some(pid) = self.child.id() {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(pid as i32),
                nix::sys::signal::Signal::SIGINT,
            );
        }
        #[cfg(not(unix))]
        let _ = self.child.start_kill();

        let status = tokio::time::timeout(Duration::from_secs(10), self.child.wait()).await;
        match status {
            Ok(Ok(_)) => {}
            Ok(Err(error)) => bail!("could not wait for record_cmd: {error}"),
            Err(_) => {
                let _ = self.child.start_kill();
                bail!("record_cmd did not stop within 10s after SIGINT");
            }
        }
        if !self.file.exists() {
            bail!("record_cmd exited without writing {}", self.file.display());
        }
        let transcript = transcribe(config, &self.file).await?;
        drop(self.dir);
        Ok(transcript)
    }
}

pub async fn transcribe(config: &VoiceConfig, file: &Path) -> Result<String> {
    let template = config
        .transcribe_cmd
        .as_deref()
        .context("voice.transcribe_cmd is not configured")?;
    let output = tokio::time::timeout(
        config.transcribe_timeout(),
        shell(&fill_file(template, file))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output(),
    )
    .await
    .map_err(|_| {
        anyhow::anyhow!(
            "transcribe_cmd did not finish within {}s",
            config.transcribe_timeout().as_secs()
        )
    })?
    .context("could not run transcribe_cmd")?;
    if !output.status.success() {
        bail!(
            "transcribe_cmd exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Speech playback in progress: one subprocess per segment, run in order.
pub struct Speaking {
    config: VoiceConfig,
    queue: std::collections::VecDeque<Segment>,
    current: Option<Child>,
}

impl Speaking {
    /// Queue a reply. Returns None when no segment has a speech command, so
    /// the caller never enters a speaking state it cannot leave.
    pub fn start(config: &VoiceConfig, text: &str) -> Option<Self> {
        let queue: std::collections::VecDeque<Segment> = segments(text)
            .into_iter()
            .filter(|s| config.speak_command(s.lang.as_deref()).is_some())
            .collect();
        if queue.is_empty() {
            return None;
        }
        Some(Self {
            config: config.clone(),
            queue,
            current: None,
        })
    }

    /// Advance playback. Returns Ok(true) while something is still playing
    /// or queued, Ok(false) once every segment has finished.
    pub async fn step(&mut self) -> Result<bool> {
        if let Some(child) = &mut self.current {
            match child.try_wait() {
                Ok(Some(_)) => self.current = None,
                Ok(None) => return Ok(true),
                Err(error) => bail!("could not wait for speak command: {error}"),
            }
        }
        let Some(segment) = self.queue.pop_front() else {
            return Ok(false);
        };
        let command = self
            .config
            .speak_command(segment.lang.as_deref())
            .expect("filtered at start")
            .to_string();
        let mut child = shell(&command)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("could not start speak command {command:?}"))?;
        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            let text = segment.text;
            tokio::spawn(async move {
                let _ = stdin.write_all(text.as_bytes()).await;
                let _ = stdin.shutdown().await;
            });
        }
        self.current = Some(child);
        Ok(true)
    }

    /// Stop playback and drop everything queued.
    pub fn interrupt(&mut self) {
        if let Some(child) = &mut self.current {
            let _ = child.start_kill();
        }
        self.current = None;
        self.queue.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(lang: Option<&str>, text: &str) -> Segment {
        Segment {
            lang: lang.map(str::to_string),
            text: text.to_string(),
        }
    }

    #[test]
    fn segments_route_tagged_spans_and_leave_other_brackets_alone() {
        let text = "Say hello: [cs]Dobrý den.[/cs] Then [cs]Jak se máte?[/cs] (see [1] and [TODO])";
        assert_eq!(
            segments(text),
            vec![
                seg(None, "Say hello:"),
                seg(Some("cs"), "Dobrý den."),
                seg(None, "Then"),
                seg(Some("cs"), "Jak se máte?"),
                seg(None, "(see [1] and [TODO])"),
            ]
        );
        assert_eq!(segments("plain"), vec![seg(None, "plain")]);
        assert_eq!(
            segments("[sv]unclosed runs on"),
            vec![seg(Some("sv"), "unclosed runs on")]
        );
        assert!(segments("   ").is_empty());
        assert_eq!(segments("[cs][/cs]nothing"), vec![seg(None, "nothing")]);
    }

    #[test]
    fn speak_command_falls_back_to_default_language() {
        let mut config = VoiceConfig::default();
        config.speak.insert("en".into(), "say-en".into());
        config.speak.insert("cs".into(), "say-cs".into());
        config.speak_default = Some("en".into());
        assert_eq!(config.speak_command(Some("cs")), Some("say-cs"));
        assert_eq!(config.speak_command(Some("sv")), Some("say-en"));
        assert_eq!(config.speak_command(None), Some("say-en"));
        config.speak_default = None;
        assert_eq!(config.speak_command(None), None);
        assert!(Speaking::start(&config, "[sv]hej[/sv]").is_none());
    }

    #[test]
    fn file_placeholder_is_shell_quoted() {
        let filled = fill_file("rec {file} {file}", Path::new("/tmp/it's here/a.wav"));
        assert_eq!(
            filled,
            "rec '/tmp/it'\\''s here/a.wav' '/tmp/it'\\''s here/a.wav'"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn record_stop_transcribe_round_trip_with_fake_commands() {
        let config = VoiceConfig {
            record_cmd: Some(
                "trap 'exit 0' INT; printf fake-audio > {file}; while :; do sleep 0.05; done"
                    .into(),
            ),
            transcribe_cmd: Some(
                "test \"$(cat {file})\" = fake-audio && echo '  dobrý den  '".into(),
            ),
            ..Default::default()
        };
        let recording = Recording::start(&config).unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;
        let text = recording.stop_and_transcribe(&config).await.unwrap();
        assert_eq!(text, "dobrý den");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn transcribe_failures_are_reported_not_swallowed() {
        let config = VoiceConfig {
            transcribe_cmd: Some("echo nope >&2; exit 3".into()),
            ..Default::default()
        };
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.wav");
        std::fs::write(&file, b"x").unwrap();
        let error = transcribe(&config, &file).await.unwrap_err().to_string();
        assert!(
            error.contains("exited") && error.contains("nope"),
            "{error}"
        );

        let slow = VoiceConfig {
            transcribe_cmd: Some("sleep 5".into()),
            transcribe_timeout_secs: 1,
            ..Default::default()
        };
        let error = transcribe(&slow, &file).await.unwrap_err().to_string();
        assert!(error.contains("did not finish"), "{error}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn speaking_runs_segments_in_order_and_interrupts() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("spoken.txt");
        let mut config = VoiceConfig::default();
        config.speak.insert(
            "en".into(),
            format!("printf 'en:%s\\n' \"$(cat)\" >> '{}'", log.display()),
        );
        config.speak.insert(
            "cs".into(),
            format!("printf 'cs:%s\\n' \"$(cat)\" >> '{}'", log.display()),
        );
        config.speak_default = Some("en".into());

        let mut speaking = Speaking::start(&config, "Hello. [cs]Ahoj.[/cs] Bye.").unwrap();
        while speaking.step().await.unwrap() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(
            std::fs::read_to_string(&log).unwrap(),
            "en:Hello.\ncs:Ahoj.\nen:Bye.\n"
        );

        config.speak.insert("en".into(), "sleep 5".into());
        let mut speaking = Speaking::start(&config, "one [cs]two[/cs]").unwrap();
        assert!(speaking.step().await.unwrap());
        speaking.interrupt();
        assert!(!speaking.step().await.unwrap());
    }
}
