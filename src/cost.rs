use crate::api::types::Usage;
use serde::{Deserialize, Serialize};

/// Model prices in USD per million tokens.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ModelPricing {
    pub input: f64,
    pub output: f64,
    #[serde(default)]
    pub cache_read: f64,
    #[serde(default)]
    pub cache_write: f64,
}

/// Tracks token usage and estimated cost for a session.
#[derive(Debug, Default)]
pub struct CostTracker {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_creation_tokens: u64,
    pub model: String,
    pricing: Option<ModelPricing>,
}

impl CostTracker {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            pricing: built_in_pricing(model),
            ..Default::default()
        }
    }

    pub fn set_pricing_override(&mut self, pricing: Option<ModelPricing>) {
        if let Some(pricing) = pricing {
            self.pricing = Some(pricing);
        }
    }

    pub fn add_usage(&mut self, usage: &Usage) {
        self.input_tokens += usage.input_tokens as u64;
        self.output_tokens += usage.output_tokens as u64;
        self.cache_read_tokens += usage.cache_read_tokens as u64;
        self.cache_creation_tokens += usage.cache_creation_tokens as u64;
    }

    /// Estimated cost in USD based on model pricing.
    pub fn total_cost_usd(&self) -> f64 {
        let Some(pricing) = self.pricing else {
            return 0.0;
        };

        let per_m = |tokens: u64, price: f64| tokens as f64 / 1_000_000.0 * price;

        per_m(self.input_tokens, pricing.input)
            + per_m(self.output_tokens, pricing.output)
            + per_m(self.cache_read_tokens, pricing.cache_read)
            + per_m(self.cache_creation_tokens, pricing.cache_write)
    }

    pub fn format_summary(&self) -> String {
        let cost = self
            .pricing
            .map(|_| format!("${:.4}", self.total_cost_usd()))
            .unwrap_or_else(|| "unavailable".to_string());
        format!(
            "Cost: {} | Tokens: {}in / {}out{}",
            cost,
            self.input_tokens,
            self.output_tokens,
            if self.cache_read_tokens > 0 {
                format!(" / {}cache", self.cache_read_tokens)
            } else {
                String::new()
            }
        )
    }
}

fn pricing(input: f64, output: f64, cache_read: f64, cache_write: f64) -> ModelPricing {
    ModelPricing {
        input,
        output,
        cache_read,
        cache_write,
    }
}

fn built_in_pricing(model: &str) -> Option<ModelPricing> {
    if model == "gpt-5.6" || model.contains("gpt-5.6-sol") {
        Some(pricing(5.0, 30.0, 0.5, 6.25))
    } else if model.contains("gpt-5.6-terra") {
        Some(pricing(2.5, 15.0, 0.25, 3.125))
    } else if model.contains("gpt-5.6-luna") {
        Some(pricing(1.0, 6.0, 0.1, 1.25))
    } else if model.contains("gpt-5.3-codex") || model.contains("gpt-5.2-codex") {
        Some(pricing(1.75, 14.0, 0.175, 2.1875))
    } else if model.contains("opus") {
        Some(pricing(15.0, 75.0, 1.5, 18.75))
    } else if model.contains("sonnet") {
        Some(pricing(3.0, 15.0, 0.3, 3.75))
    } else if model.contains("haiku") {
        Some(pricing(0.25, 1.25, 0.025, 0.3))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tracker_is_zero() {
        let tracker = CostTracker::new("claude-sonnet-4-20250514");
        assert_eq!(tracker.input_tokens, 0);
        assert_eq!(tracker.output_tokens, 0);
        assert!((tracker.total_cost_usd() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn add_usage_accumulates() {
        let mut tracker = CostTracker::new("claude-sonnet-4-20250514");
        tracker.add_usage(&Usage {
            input_tokens: 1000,
            output_tokens: 500,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        });
        tracker.add_usage(&Usage {
            input_tokens: 2000,
            output_tokens: 300,
            cache_read_tokens: 100,
            cache_creation_tokens: 0,
        });
        assert_eq!(tracker.input_tokens, 3000);
        assert_eq!(tracker.output_tokens, 800);
        assert_eq!(tracker.cache_read_tokens, 100);
    }

    #[test]
    fn sonnet_pricing() {
        let mut tracker = CostTracker::new("claude-sonnet-4-20250514");
        tracker.add_usage(&Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        });
        // sonnet: $3/M input + $15/M output = $18
        assert!((tracker.total_cost_usd() - 18.0).abs() < 0.01);
    }

    #[test]
    fn opus_pricing() {
        let mut tracker = CostTracker::new("claude-opus-4-20250514");
        tracker.add_usage(&Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        });
        // opus: $15/M input + $75/M output = $90
        assert!((tracker.total_cost_usd() - 90.0).abs() < 0.01);
    }

    #[test]
    fn haiku_pricing() {
        let mut tracker = CostTracker::new("claude-haiku-4-5-20251001");
        tracker.add_usage(&Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        });
        // haiku: $0.25/M input + $1.25/M output = $1.50
        assert!((tracker.total_cost_usd() - 1.50).abs() < 0.01);
    }

    #[test]
    fn unknown_model_reports_unavailable_pricing() {
        let mut tracker = CostTracker::new("some-future-model");
        tracker.add_usage(&Usage {
            input_tokens: 1_000_000,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        });
        assert_eq!(tracker.total_cost_usd(), 0.0);
        assert!(tracker.format_summary().contains("unavailable"));
    }

    #[test]
    fn gpt_5_6_sol_pricing() {
        let mut tracker = CostTracker::new("gpt-5.6-sol");
        tracker.add_usage(&Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        });
        assert!((tracker.total_cost_usd() - 35.0).abs() < 0.01);
    }

    #[test]
    fn config_pricing_overrides_built_in_value() {
        let mut tracker = CostTracker::new("gpt-5.6-sol");
        tracker.set_pricing_override(Some(ModelPricing {
            input: 1.0,
            output: 2.0,
            cache_read: 0.0,
            cache_write: 0.0,
        }));
        tracker.add_usage(&Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        });
        assert!((tracker.total_cost_usd() - 3.0).abs() < 0.01);
    }

    #[test]
    fn cache_tokens_affect_cost() {
        let mut tracker = CostTracker::new("claude-sonnet-4-20250514");
        tracker.add_usage(&Usage {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 1_000_000,
            cache_creation_tokens: 1_000_000,
        });
        // sonnet cache: $0.3/M read + $3.75/M write = $4.05
        assert!((tracker.total_cost_usd() - 4.05).abs() < 0.01);
    }

    #[test]
    fn format_summary_includes_tokens() {
        let mut tracker = CostTracker::new("claude-sonnet-4-20250514");
        tracker.add_usage(&Usage {
            input_tokens: 500,
            output_tokens: 200,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        });
        let summary = tracker.format_summary();
        assert!(summary.contains("500in"));
        assert!(summary.contains("200out"));
        assert!(summary.contains("$"));
    }

    #[test]
    fn format_summary_shows_cache_when_present() {
        let mut tracker = CostTracker::new("claude-sonnet-4-20250514");
        tracker.add_usage(&Usage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_tokens: 300,
            cache_creation_tokens: 0,
        });
        let summary = tracker.format_summary();
        assert!(summary.contains("300cache"));
    }

    #[test]
    fn format_summary_hides_cache_when_zero() {
        let mut tracker = CostTracker::new("claude-sonnet-4-20250514");
        tracker.add_usage(&Usage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        });
        let summary = tracker.format_summary();
        assert!(!summary.contains("cache"));
    }
}
