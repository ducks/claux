use anyhow::Result;
use std::process::Command;
use tracing::warn;

/// A plugin that can inject context into the system prompt.
pub trait Plugin: Send + Sync {
    /// Returns the name of the plugin.
    fn name(&self) -> &str;

    /// Executes the plugin and returns context text to append to the system prompt.
    /// Returns None if the plugin has nothing to contribute.
    fn execute(&self) -> Result<Option<String>>;
}

/// A plugin that runs an external command and captures its output.
pub struct CommandPlugin {
    name: String,
    command: String,
    args: Vec<String>,
}

impl CommandPlugin {
    pub fn new(name: &str, command: &str, args: &[String]) -> Self {
        Self {
            name: name.to_string(),
            command: command.to_string(),
            args: args.to_vec(),
        }
    }
}

impl Plugin for CommandPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn execute(&self) -> Result<Option<String>> {
        let output = Command::new(&self.command)
            .args(&self.args)
            .output()?;

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

    /// Executes all plugins and returns combined context.
    pub fn execute_all(&self) -> Result<String> {
        let mut parts = Vec::new();

        for plugin in &self.plugins {
            match plugin.execute() {
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

    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    pub fn len(&self) -> usize {
        self.plugins.len()
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

    #[test]
    fn test_command_plugin_success() {
        let plugin = CommandPlugin::new("echo-test", "echo", &vec!["hello world".to_string()]);
        let result = plugin.execute().unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "hello world");
    }

    #[test]
    fn test_command_plugin_empty_output() {
        let plugin = CommandPlugin::new("true", "true", &vec![]);
        let result = plugin.execute().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_command_plugin_failure() {
        let plugin = CommandPlugin::new("fail", "false", &vec![]);
        let result = plugin.execute().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_plugin_registry_empty() {
        let registry = PluginRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        let output = registry.execute_all().unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_plugin_registry_multiple() {
        let mut registry = PluginRegistry::new();
        let args1: Vec<String> = vec!["first".to_string()];
        let args2: Vec<String> = vec!["second".to_string()];
        registry.add(Box::new(CommandPlugin::new("echo1", "echo", &args1)));
        registry.add(Box::new(CommandPlugin::new("echo2", "echo", &args2)));

        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 2);

        let output = registry.execute_all().unwrap();
        assert!(output.contains("Plugin: echo1"));
        assert!(output.contains("first"));
        assert!(output.contains("Plugin: echo2"));
        assert!(output.contains("second"));
    }

    #[test]
    fn test_plugin_name() {
        let plugin = CommandPlugin::new("my-plugin", "echo", &vec![]);
        assert_eq!(plugin.name(), "my-plugin");
    }
}
