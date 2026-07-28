use crate::api::types::Usage;
pub use crate::model::ModelPricing;

/// Tracks token usage and estimated cost for a session.
#[derive(Debug, Default)]
pub struct CostTracker {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_creation_tokens: u64,
    provider_cost_usd: Option<f64>,
    pricing: Option<ModelPricing>,
}

impl CostTracker {
    pub fn new(model: &str) -> Self {
        Self {
            pricing: crate::model::built_in_metadata(model).pricing,
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
        if let Some(cost) = usage
            .provider_cost_usd
            .filter(|cost| cost.is_finite() && *cost >= 0.0)
        {
            *self.provider_cost_usd.get_or_insert(0.0) += cost;
        }
    }

    /// Clear session usage while preserving the model's resolved pricing.
    pub fn reset_usage(&mut self) {
        self.input_tokens = 0;
        self.output_tokens = 0;
        self.cache_read_tokens = 0;
        self.cache_creation_tokens = 0;
        self.provider_cost_usd = None;
    }

    /// Actual provider-reported cost when available, otherwise an estimate.
    pub fn total_cost_usd(&self) -> f64 {
        if let Some(cost) = self.provider_cost_usd {
            return cost;
        }
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
            .provider_cost_usd
            .map(format_cost)
            .or_else(|| self.pricing.map(|_| format_cost(self.total_cost_usd())))
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

fn format_cost(cost: f64) -> String {
    if cost > 0.0 && cost < 0.0001 {
        format!("${cost:.8}")
    } else {
        format!("${cost:.4}")
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
            provider_cost_usd: None,
        });
        tracker.add_usage(&Usage {
            input_tokens: 2000,
            output_tokens: 300,
            cache_read_tokens: 100,
            cache_creation_tokens: 0,
            provider_cost_usd: None,
        });
        assert_eq!(tracker.input_tokens, 3000);
        assert_eq!(tracker.output_tokens, 800);
        assert_eq!(tracker.cache_read_tokens, 100);
    }

    #[test]
    fn reset_usage_preserves_pricing() {
        let mut tracker = CostTracker::new("private-model");
        tracker.set_pricing_override(Some(ModelPricing {
            input: 2.0,
            output: 4.0,
            cache_read: 0.5,
            cache_write: 1.0,
        }));
        tracker.add_usage(&Usage {
            input_tokens: 1_000_000,
            output_tokens: 500_000,
            cache_read_tokens: 100,
            cache_creation_tokens: 50,
            provider_cost_usd: None,
        });

        tracker.reset_usage();

        assert_eq!(tracker.input_tokens, 0);
        assert_eq!(tracker.output_tokens, 0);
        assert_eq!(tracker.cache_read_tokens, 0);
        assert_eq!(tracker.cache_creation_tokens, 0);
        assert_eq!(tracker.total_cost_usd(), 0.0);

        tracker.add_usage(&Usage {
            input_tokens: 1_000_000,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            provider_cost_usd: None,
        });
        assert_eq!(tracker.total_cost_usd(), 2.0);
    }

    #[test]
    fn sonnet_pricing() {
        let mut tracker = CostTracker::new("claude-sonnet-4-20250514");
        tracker.add_usage(&Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            provider_cost_usd: None,
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
            provider_cost_usd: None,
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
            provider_cost_usd: None,
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
            provider_cost_usd: None,
        });
        assert_eq!(tracker.total_cost_usd(), 0.0);
        assert!(tracker.format_summary().contains("unavailable"));
    }

    #[test]
    fn provider_reported_cost_works_without_known_pricing_and_accumulates() {
        let mut tracker = CostTracker::new("openrouter/unknown-model");
        tracker.add_usage(&Usage {
            input_tokens: 100,
            output_tokens: 20,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            provider_cost_usd: Some(0.0012),
        });
        tracker.add_usage(&Usage {
            input_tokens: 200,
            output_tokens: 40,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            provider_cost_usd: Some(0.0034),
        });

        assert!((tracker.total_cost_usd() - 0.0046).abs() < f64::EPSILON);
        assert!(tracker.format_summary().contains("$0.0046"));
    }

    #[test]
    fn provider_reported_cost_takes_precedence_over_estimate() {
        let mut tracker = CostTracker::new("claude-sonnet-4-20250514");
        tracker.add_usage(&Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            provider_cost_usd: Some(1.25),
        });

        assert_eq!(tracker.total_cost_usd(), 1.25);
    }

    #[test]
    fn tiny_reported_cost_remains_visible() {
        let mut tracker = CostTracker::new("unknown-model");
        tracker.add_usage(&Usage {
            input_tokens: 10,
            output_tokens: 2,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            provider_cost_usd: Some(0.00003125),
        });

        assert!(tracker.format_summary().contains("$0.00003125"));
    }

    #[test]
    fn reset_usage_clears_provider_reported_cost() {
        let mut tracker = CostTracker::new("unknown-model");
        tracker.add_usage(&Usage {
            input_tokens: 100,
            output_tokens: 20,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            provider_cost_usd: Some(0.0012),
        });

        tracker.reset_usage();

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
            provider_cost_usd: None,
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
            provider_cost_usd: None,
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
            provider_cost_usd: None,
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
            provider_cost_usd: None,
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
            provider_cost_usd: None,
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
            provider_cost_usd: None,
        });
        let summary = tracker.format_summary();
        assert!(!summary.contains("cache"));
    }
}
