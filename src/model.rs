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

/// Runtime properties shared by context management and cost tracking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelMetadata {
    pub context_window: usize,
    pub pricing: Option<ModelPricing>,
}

impl ModelMetadata {
    pub fn with_overrides(
        mut self,
        context_window: Option<usize>,
        pricing: Option<ModelPricing>,
    ) -> Self {
        if let Some(context_window) = context_window.filter(|window| *window > 0) {
            self.context_window = context_window;
        }
        if let Some(pricing) = pricing {
            self.pricing = Some(pricing);
        }
        self
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

/// Built-in model knowledge. Keep family matching here so adding a model
/// cannot update context management while accidentally omitting cost tracking.
pub fn built_in_metadata(model: &str) -> ModelMetadata {
    let (context_window, pricing) = if model == "gpt-5.6" || model.contains("gpt-5.6-sol") {
        (1_050_000, Some(pricing(5.0, 30.0, 0.5, 6.25)))
    } else if model.contains("gpt-5.6-terra") {
        (1_050_000, Some(pricing(2.5, 15.0, 0.25, 3.125)))
    } else if model.contains("gpt-5.6-luna") {
        (1_050_000, Some(pricing(1.0, 6.0, 0.1, 1.25)))
    } else if model.contains("gpt-5.6") {
        (1_050_000, None)
    } else if model.contains("gpt-5.3-codex") || model.contains("gpt-5.2-codex") {
        (400_000, Some(pricing(1.75, 14.0, 0.175, 2.1875)))
    } else if model.contains("gpt-5.1-codex") || model.contains("gpt-5-codex") {
        (400_000, None)
    } else if model.contains("opus") {
        (200_000, Some(pricing(15.0, 75.0, 1.5, 18.75)))
    } else if model.contains("sonnet") {
        (200_000, Some(pricing(3.0, 15.0, 0.3, 3.75)))
    } else if model.contains("haiku") {
        (200_000, Some(pricing(0.25, 1.25, 0.025, 0.3)))
    } else if model.contains("gpt-4o") || model.contains("gpt-4") {
        (128_000, None)
    } else if model.contains("gpt-3.5") {
        (16_000, None)
    } else {
        // Conservative default for unknown models.
        (128_000, None)
    };

    ModelMetadata {
        context_window,
        pricing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_family_resolves_context_and_pricing_together() {
        let metadata = built_in_metadata("openrouter/openai/gpt-5.6-terra");
        assert_eq!(metadata.context_window, 1_050_000);
        assert_eq!(metadata.pricing.unwrap().input, 2.5);
    }

    #[test]
    fn overrides_take_precedence_and_zero_context_is_ignored() {
        let custom = pricing(0.5, 1.5, 0.0, 0.0);
        let metadata = built_in_metadata("unknown")
            .with_overrides(Some(64_000), Some(custom))
            .with_overrides(Some(0), None);
        assert_eq!(metadata.context_window, 64_000);
        assert_eq!(metadata.pricing, Some(custom));
    }
}
