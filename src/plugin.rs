use anyhow::Result;
use std::collections::HashMap;
use std::process::Command;
use tracing::warn;

use crate::config::HookTrigger;

/// A plugin that can inject context into the system prompt or react to events.
pub trait Plugin: Send + Sync {
    /// Returns the name of the plugin.
    fn name(&self) -> &str;

    /// Returns the trigger for this plugin.
    fn trigger(&self) -> &HookTrigger;

    /// Executes the plugin with optional environment variables.
    /// Returns None if the plugin has nothing to contribute.
    fn execute(&self, env_vars: Option<&HashMap<String, String>>) -> Result<Option<String>>;
}

/// A plugin that runs an external command and captures its output.
pub struct CommandPlugin {
    name: String,
    command: String,
    args: Vec<String>,
    trigger: HookTrigger,
}

impl CommandPlugin {
    pub fn new(name: &str, command: &str, args: &[String], trigger: HookTrigger) -> Self {
        Self {
            name: name.to_string(),
            command: command.to_string(),
            args: args.to_vec(),
            trigger,
        }
    }

    pub fn trigger(&self) -> &HookTrigger {
        &self.trigger
    }
}

impl Plugin for CommandPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn trigger(&self) -> &HookTrigger {
        &self.trigger
    }

    fn execute(&self, env_vars: Option<&HashMap<String, String>>) -> Result<Option<String>> {
        let mut cmd = Command::new(&self.command);
        cmd.args(&self.args);

        // Inject environment variables if provided
        if let Some(env) = env_vars {
            for (key, value) in env {
                cmd.env(key, value);
            }
        }

        let output = cmd.output()?;

        if !output.status.success() {
            warn!("Plugin '{}' failed: {}", self.name, String::from_utf8_lossy(&output.stderr));
            return Ok(None);
        }

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if stdout.is_empty() {
            Ok(None)
        } else {
            Ok(Some(stdout))
        }
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
    pub fn execute_all(&self, trigger: &HookTrigger, env_vars: Option<&HashMap<String, String>>) -> Result<String> {
        let mut parts = Vec::new();

        for plugin in &self.plugins {
            if plugin.trigger() != trigger {
                continue;
            }

            match plugin.execute(env_vars) {
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
    pub fn execute_side_effects(&self, trigger: &HookTrigger, env_vars: Option<&HashMap<String, String>>) -> Result<()> {
        for plugin in &self.plugins {
            if plugin.trigger() != trigger {
                continue;
            }

            if let Err(e) = plugin.execute(env_vars) {
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
        self.plugins.iter().filter(|p| p.trigger() == trigger).count()
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

    #[test]
    fn test_command_plugin_success() {
        let plugin = CommandPlugin::new("echo-test", "echo", &vec!["hello world".to_string()], HookTrigger::OnContextBuild);
        let result = plugin.execute(None).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "hello world");
    }

    #[test]
    fn test_command_plugin_empty_output() {
        let plugin = CommandPlugin::new("true", "true", &vec![], HookTrigger::OnContextBuild);
        let result = plugin.execute(None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_command_plugin_failure() {
        let plugin = CommandPlugin::new("fail", "false", &vec![], HookTrigger::OnContextBuild);
        let result = plugin.execute(None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_plugin_registry_empty() {
        let registry = PluginRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        let output = registry.execute_all(&HookTrigger::OnContextBuild, None).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_plugin_registry_multiple() {
        let mut registry = PluginRegistry::new();
        let args1: Vec<String> = vec!["first".to_string()];
        let args2: Vec<String> = vec!["second".to_string()];
        registry.add(Box::new(CommandPlugin::new("echo1", "echo", &args1, HookTrigger::OnContextBuild)));
        registry.add(Box::new(CommandPlugin::new("echo2", "echo", &args2, HookTrigger::OnContextBuild)));

        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 2);

        let output = registry.execute_all(&HookTrigger::OnContextBuild, None).unwrap();
        assert!(output.contains("Plugin: echo1"));
        assert!(output.contains("first"));
        assert!(output.contains("Plugin: echo2"));
        assert!(output.contains("second"));
    }

    #[test]
    fn test_plugin_registry_filter_by_trigger() {
        let mut registry = PluginRegistry::new();
        let args1: Vec<String> = vec!["context".to_string()];
        let args2: Vec<String> = vec!["tool".to_string()];
        registry.add(Box::new(CommandPlugin::new("ctx", "echo", &args1, HookTrigger::OnContextBuild)));
        registry.add(Box::new(CommandPlugin::new("tool", "echo", &args2, HookTrigger::OnToolStart)));

        let ctx_output = registry.execute_all(&HookTrigger::OnContextBuild, None).unwrap();
        let tool_output = registry.execute_all(&HookTrigger::OnToolStart, None).unwrap();

        assert!(ctx_output.contains("ctx"));
        assert!(!ctx_output.contains("tool"));
        assert!(tool_output.contains("tool"));
        assert!(!tool_output.contains("ctx"));
    }

    #[test]
    fn test_plugin_name() {
        let plugin = CommandPlugin::new("my-plugin", "echo", &vec![], HookTrigger::OnContextBuild);
        assert_eq!(plugin.name(), "my-plugin");
    }

    #[test]
    fn test_plugin_trigger() {
        let plugin = CommandPlugin::new("my-plugin", "echo", &vec![], HookTrigger::OnToolStart);
        assert_eq!(*plugin.trigger(), HookTrigger::OnToolStart);
    }

    #[test]
    fn test_plugin_with_env_vars() {
        let mut env = HashMap::new();
        env.insert("TEST_VAR".to_string(), "test_value".to_string());
        
        let plugin = CommandPlugin::new("env-test", "sh", &vec!["-c".to_string(), "echo $TEST_VAR".to_string()], HookTrigger::OnContextBuild);
        let result = plugin.execute(Some(&env)).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "test_value");
    }
}
