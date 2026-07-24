use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::process::Stdio;
use tracing::warn;

use crate::config::HookTrigger;

const HOOK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
const HOOK_OUTPUT_LIMIT: usize = 64 * 1024;

/// A plugin that can inject context into the system prompt or react to events.
#[async_trait]
pub trait Plugin: Send + Sync {
    /// Returns the name of the plugin.
    fn name(&self) -> &str;

    /// Returns the trigger for this plugin.
    fn trigger(&self) -> &HookTrigger;

    /// Executes the plugin with optional environment variables.
    /// Returns None if the plugin has nothing to contribute.
    async fn execute(&self, env_vars: Option<&HashMap<String, String>>) -> Result<Option<String>>;
}

/// A plugin that runs an external command and captures its output.
pub struct CommandPlugin {
    name: String,
    command: String,
    args: Vec<String>,
    trigger: HookTrigger,
    timeout: std::time::Duration,
}

impl CommandPlugin {
    pub fn new(name: &str, command: &str, args: &[String], trigger: HookTrigger) -> Self {
        Self {
            name: name.to_string(),
            command: command.to_string(),
            args: args.to_vec(),
            trigger,
            timeout: HOOK_TIMEOUT,
        }
    }

    pub fn trigger(&self) -> &HookTrigger {
        &self.trigger
    }

    #[cfg(test)]
    fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[async_trait]
impl Plugin for CommandPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn trigger(&self) -> &HookTrigger {
        &self.trigger
    }

    async fn execute(&self, env_vars: Option<&HashMap<String, String>>) -> Result<Option<String>> {
        let mut cmd = tokio::process::Command::new(&self.command);
        cmd.args(&self.args);
        cmd.stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        // Inject environment variables if provided
        if let Some(env) = env_vars {
            for (key, value) in env {
                cmd.env(key, value);
            }
        }

        let mut child = cmd.spawn()?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("failed to capture hook stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow::anyhow!("failed to capture hook stderr"))?;

        let execution = async {
            let (stdout, stderr, status) =
                tokio::try_join!(read_capped(stdout), read_capped(stderr), async {
                    child.wait().await
                })?;
            Ok::<_, std::io::Error>((stdout, stderr, status))
        };
        let (stdout, stderr, status) = tokio::time::timeout(self.timeout, execution)
            .await
            .map_err(|_| anyhow::anyhow!("hook timed out after {:?}", self.timeout))??;

        if !status.success() {
            warn!(
                "Plugin '{}' failed: {}",
                self.name,
                String::from_utf8_lossy(&stderr)
            );
            return Ok(None);
        }

        let stdout = String::from_utf8_lossy(&stdout).trim().to_string();
        if stdout.is_empty() {
            Ok(None)
        } else {
            Ok(Some(stdout))
        }
    }
}

async fn read_capped<R: tokio::io::AsyncRead + Unpin>(mut reader: R) -> std::io::Result<Vec<u8>> {
    use tokio::io::AsyncReadExt as _;

    let mut output = Vec::new();
    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let read = reader.read(&mut chunk).await?;
        if read == 0 {
            return Ok(output);
        }
        let remaining = HOOK_OUTPUT_LIMIT.saturating_sub(output.len());
        output.extend_from_slice(&chunk[..read.min(remaining)]);
    }
}

/// Registry for managing plugins.
pub struct PluginRegistry {
    plugins: Vec<Box<dyn Plugin>>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    pub fn add(&mut self, plugin: Box<dyn Plugin>) {
        self.plugins.push(plugin);
    }

    /// Executes all plugins for a specific trigger and returns combined context.
    pub async fn execute_all(
        &self,
        trigger: &HookTrigger,
        env_vars: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        let mut parts = Vec::new();

        let plugins: Vec<&Box<dyn Plugin>> = self
            .plugins
            .iter()
            .filter(|plugin| plugin.trigger() == trigger)
            .collect();
        let results =
            futures_util::future::join_all(plugins.iter().map(|plugin| plugin.execute(env_vars)))
                .await;

        for (plugin, result) in plugins.into_iter().zip(results) {
            match result {
                Ok(Some(context)) => {
                    parts.push(format!("# Plugin: {}\n{}", plugin.name(), context));
                }
                Ok(None) => {}
                Err(e) => {
                    warn!("Plugin '{}' error: {}", plugin.name(), e);
                }
            }
        }

        Ok(parts.join("\n\n"))
    }

    /// Execute plugins that don't return context (side-effect only, like logging).
    pub async fn execute_side_effects(
        &self,
        trigger: &HookTrigger,
        env_vars: Option<&HashMap<String, String>>,
    ) -> Result<()> {
        let plugins: Vec<&Box<dyn Plugin>> = self
            .plugins
            .iter()
            .filter(|plugin| plugin.trigger() == trigger)
            .collect();
        let results =
            futures_util::future::join_all(plugins.iter().map(|plugin| plugin.execute(env_vars)))
                .await;

        for (plugin, result) in plugins.into_iter().zip(results) {
            if let Err(e) = result {
                warn!("Plugin '{}' error: {}", plugin.name(), e);
            }
        }
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Get plugins for a specific trigger
    pub fn get_by_trigger(&self, trigger: &HookTrigger) -> usize {
        self.plugins
            .iter()
            .filter(|p| p.trigger() == trigger)
            .count()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_command_plugin_success() {
        let plugin = CommandPlugin::new(
            "echo-test",
            "echo",
            &["hello world".to_string()],
            HookTrigger::OnContextBuild,
        );
        let result = plugin.execute(None).await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "hello world");
    }

    #[tokio::test]
    async fn test_command_plugin_empty_output() {
        let plugin = CommandPlugin::new("true", "true", &[], HookTrigger::OnContextBuild);
        let result = plugin.execute(None).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_command_plugin_failure() {
        let plugin = CommandPlugin::new("fail", "false", &[], HookTrigger::OnContextBuild);
        let result = plugin.execute(None).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_plugin_registry_empty() {
        let registry = PluginRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        let output = registry
            .execute_all(&HookTrigger::OnContextBuild, None)
            .await
            .unwrap();
        assert!(output.is_empty());
    }

    #[tokio::test]
    async fn test_plugin_registry_multiple() {
        let mut registry = PluginRegistry::new();
        let args1: Vec<String> = vec!["first".to_string()];
        let args2: Vec<String> = vec!["second".to_string()];
        registry.add(Box::new(CommandPlugin::new(
            "echo1",
            "echo",
            &args1,
            HookTrigger::OnContextBuild,
        )));
        registry.add(Box::new(CommandPlugin::new(
            "echo2",
            "echo",
            &args2,
            HookTrigger::OnContextBuild,
        )));

        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 2);

        let output = registry
            .execute_all(&HookTrigger::OnContextBuild, None)
            .await
            .unwrap();
        assert!(output.contains("Plugin: echo1"));
        assert!(output.contains("first"));
        assert!(output.contains("Plugin: echo2"));
        assert!(output.contains("second"));
    }

    #[tokio::test]
    async fn test_plugin_registry_filter_by_trigger() {
        let mut registry = PluginRegistry::new();
        let args1: Vec<String> = vec!["context".to_string()];
        let args2: Vec<String> = vec!["tool".to_string()];
        registry.add(Box::new(CommandPlugin::new(
            "ctx",
            "echo",
            &args1,
            HookTrigger::OnContextBuild,
        )));
        registry.add(Box::new(CommandPlugin::new(
            "tool",
            "echo",
            &args2,
            HookTrigger::OnToolStart,
        )));

        let ctx_output = registry
            .execute_all(&HookTrigger::OnContextBuild, None)
            .await
            .unwrap();
        let tool_output = registry
            .execute_all(&HookTrigger::OnToolStart, None)
            .await
            .unwrap();

        assert!(ctx_output.contains("ctx"));
        assert!(!ctx_output.contains("tool"));
        assert!(tool_output.contains("tool"));
        assert!(!tool_output.contains("ctx"));
    }

    #[test]
    fn test_plugin_name() {
        let plugin = CommandPlugin::new("my-plugin", "echo", &[], HookTrigger::OnContextBuild);
        assert_eq!(plugin.name(), "my-plugin");
    }

    #[test]
    fn test_plugin_trigger() {
        let plugin = CommandPlugin::new("my-plugin", "echo", &[], HookTrigger::OnToolStart);
        assert_eq!(*plugin.trigger(), HookTrigger::OnToolStart);
    }

    #[tokio::test]
    async fn test_plugin_with_env_vars() {
        let mut env = HashMap::new();
        env.insert("TEST_VAR".to_string(), "test_value".to_string());

        let plugin = CommandPlugin::new(
            "env-test",
            "sh",
            &["-c".to_string(), "echo $TEST_VAR".to_string()],
            HookTrigger::OnContextBuild,
        );
        let result = plugin.execute(Some(&env)).await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "test_value");
    }

    #[tokio::test]
    async fn command_plugin_times_out() {
        let plugin = CommandPlugin::new(
            "slow",
            "sh",
            &["-c".to_string(), "sleep 2".to_string()],
            HookTrigger::OnToolStart,
        )
        .with_timeout(std::time::Duration::from_millis(50));

        let started = std::time::Instant::now();
        let error = plugin.execute(None).await.unwrap_err();

        assert!(error.to_string().contains("timed out"));
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }

    #[tokio::test]
    async fn command_plugin_caps_captured_output() {
        let plugin = CommandPlugin::new(
            "loud",
            "sh",
            &["-c".to_string(), "yes x | head -c 100000".to_string()],
            HookTrigger::OnContextBuild,
        );

        let output = plugin.execute(None).await.unwrap().unwrap();

        assert!(output.len() <= HOOK_OUTPUT_LIMIT);
        assert!(output.len() > HOOK_OUTPUT_LIMIT / 2);
    }
}
