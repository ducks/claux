#[cfg(unix)]
use anyhow::Context;
use anyhow::Result;
use tokio_util::sync::CancellationToken;

const INTERRUPTED_MESSAGE: &str = "Interrupted by shutdown signal.";

/// Return a cancellation token that is tripped by the terminal interrupt or
/// process termination signals used by supervisors such as GNU timeout.
pub fn one_shot_cancellation_token() -> Result<CancellationToken> {
    let token = CancellationToken::new();
    let shutdown = token.clone();

    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .context("could not install SIGTERM handler")?;
        tokio::spawn(async move {
            tokio::select! {
                result = tokio::signal::ctrl_c() => {
                    if let Err(error) = result {
                        tracing::warn!("could not listen for Ctrl+C: {error}");
                        return;
                    }
                }
                _ = terminate.recv() => {}
            }
            shutdown.cancel();
        });
    }

    #[cfg(not(unix))]
    tokio::spawn(async move {
        match tokio::signal::ctrl_c().await {
            Ok(()) => shutdown.cancel(),
            Err(error) => tracing::warn!("could not listen for Ctrl+C: {error}"),
        }
    });

    Ok(token)
}

/// Cancellation ends the engine loop cleanly so its partial messages and tool
/// trace remain available. Convert that clean return into an error only after
/// the caller has regained ownership of the engine and can export them.
pub fn classify_one_shot_response<T>(response: Result<T>, cancelled: bool) -> Result<T> {
    if cancelled {
        anyhow::bail!(INTERRUPTED_MESSAGE);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelled_one_shot_becomes_an_error_for_transcript_export() {
        let error = classify_one_shot_response(Ok("partial"), true).unwrap_err();
        assert_eq!(error.to_string(), INTERRUPTED_MESSAGE);
    }

    #[test]
    fn completed_one_shot_preserves_its_result() {
        assert_eq!(
            classify_one_shot_response(Ok("complete"), false).unwrap(),
            "complete"
        );
    }

    #[test]
    fn ordinary_errors_are_preserved() {
        let error =
            classify_one_shot_response::<()>(Err(anyhow::anyhow!("provider failed")), false)
                .unwrap_err();
        assert_eq!(error.to_string(), "provider failed");
    }
}
