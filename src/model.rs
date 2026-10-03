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

/// Exact-ID baseline estimates, checked 2026-10-03 against provider model/pricing docs.
/// Standard short-context prices; provider charges and configured metadata take precedence.
/// Unknown IDs deliberately receive no price and a conservative context window.
pub fn built_in_metadata(model: &str) -> ModelMetadata {
    let model = model.strip_prefix("openrouter/").unwrap_or(model);
    let model = model
        .strip_prefix("openai/")
        .or_else(|| model.strip_prefix("anthropic/"))
        .unwrap_or(model);
    let (context_window, pricing) = match model {
        "gpt-5.6" | "gpt-5.6-sol" => (1_050_000, Some(pricing(4.0, 20.0, 0.4, 5.0))),
        "gpt-5.6-terra" => (1_050_000, Some(pricing(2.0, 12.0, 0.2, 2.5))),
        "gpt-5.6-luna" => (1_050_000, Some(pricing(0.2, 1.2, 0.02, 0.25))),
        "gpt-5.3-codex" | "gpt-5.2-codex" | "gpt-5.1-codex" | "gpt-5-codex" => (400_000, None),
        "claude-opus-4-20250514" | "claude-opus-4-1-20250805" => {
            (200_000, Some(pricing(15.0, 75.0, 1.5, 18.75)))
        }
        "claude-sonnet-4-20250514" | "claude-sonnet-4-5-20250929" => {
            (200_000, Some(pricing(3.0, 15.0, 0.3, 3.75)))
        }
        "claude-haiku-4-5-20251001" | "claude-haiku-4-5" => {
            (200_000, Some(pricing(1.0, 5.0, 0.1, 1.25)))
        }
        "gpt-4.1" | "gpt-4.1-2025-04-14" => (1_047_576, Some(pricing(2.0, 8.0, 0.5, 0.0))),
        _ => (128_000, None),
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
        assert_eq!(metadata.pricing.unwrap().input, 2.0);
    }

    #[test]
    fn unknown_names_do_not_inherit_family_prices() {
        for model in [
            "my-opus",
            "future-sonnet",
            "not-gpt-5.6-sol",
            "gpt-4.1-custom",
        ] {
            assert_eq!(built_in_metadata(model).pricing, None);
            assert_eq!(built_in_metadata(model).context_window, 128_000);
        }
        assert_eq!(built_in_metadata("gpt-4.1").context_window, 1_047_576);
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
