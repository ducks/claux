use crate::cost::{CostTracker, UsageSummary};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct OneShotOutput<'a> {
    pub schema_version: u8,
    pub result: &'a str,
    pub model: &'a str,
    pub usage: UsageSummary,
}

impl<'a> OneShotOutput<'a> {
    pub fn new(result: &'a str, model: &'a str, cost: &CostTracker) -> Self {
        Self {
            schema_version: 1,
            result,
            model,
            usage: cost.usage_summary(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::Usage;

    #[test]
    fn serializes_stable_one_shot_contract() {
        let mut cost = CostTracker::new("unknown-model");
        cost.add_usage(&Usage {
            input_tokens: 12,
            output_tokens: 4,
            cache_read_tokens: 8,
            cache_creation_tokens: 2,
            provider_cost_usd: Some(0.00042),
        });

        let value = serde_json::to_value(OneShotOutput::new("done", "test/model", &cost)).unwrap();

        assert_eq!(
            value,
            serde_json::json!({
                "schema_version": 1,
                "result": "done",
                "model": "test/model",
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 4,
                    "cache_read_tokens": 8,
                    "cache_creation_tokens": 2,
                    "cost_usd": 0.00042
                }
            })
        );
    }
}
