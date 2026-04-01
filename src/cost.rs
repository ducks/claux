use crate::api::types::Usage;

/// Tracks token usage and estimated cost for a session.
#[derive(Debug, Default)]
pub struct CostTracker {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_creation_tokens: u64,
    pub model: String,
}

impl CostTracker {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            ..Default::default()
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
        let (input_price, output_price, cache_read_price, cache_write_price) =
            model_pricing(&self.model);

        let per_m = |tokens: u64, price: f64| tokens as f64 / 1_000_000.0 * price;

        per_m(self.input_tokens, input_price)
            + per_m(self.output_tokens, output_price)
            + per_m(self.cache_read_tokens, cache_read_price)
            + per_m(self.cache_creation_tokens, cache_write_price)
    }

    pub fn format_summary(&self) -> String {
        format!(
            "Cost: ${:.4} | Tokens: {}in / {}out{}",
            self.total_cost_usd(),
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

/// Returns (input $/M, output $/M, cache_read $/M, cache_write $/M).
fn model_pricing(model: &str) -> (f64, f64, f64, f64) {
    if model.contains("opus") {
        (15.0, 75.0, 1.5, 18.75)
    } else if model.contains("sonnet") {
        (3.0, 15.0, 0.3, 3.75)
    } else if model.contains("haiku") {
        (0.25, 1.25, 0.025, 0.3)
    } else {
        // Unknown model, use sonnet pricing as default
        (3.0, 15.0, 0.3, 3.75)
    }
}
