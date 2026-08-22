use serde::{Deserialize, Serialize};

/// A message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

/// Message content can be a simple string or an array of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A content block within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },

    /// Base64-encoded image input. The nested source shape is accepted by
    /// Anthropic directly and translated by the OpenAI-family adapters.
    #[serde(rename = "image")]
    Image { source: ImageSource },

    /// Provider reasoning state retained for multi-round tool use. This is
    /// deliberately not rendered as assistant text.
    #[serde(rename = "reasoning")]
    Reasoning {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        details: Vec<serde_json::Value>,
    },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },

    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

/// Tool definition sent to the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Token usage from the API.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    /// Fresh, non-cached input tokens. Cache reads and writes are tracked in
    /// their own mutually exclusive fields.
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: u32,
    pub cache_creation_tokens: u32,
    /// Exact charge reported by the provider for this API request.
    pub provider_cost_usd: Option<f64>,
}

impl Usage {
    /// Normalize OpenAI usage, whose top-level input count includes cached
    /// tokens, into the mutually exclusive token classes Claux uses for cost
    /// and context accounting.
    pub fn from_openai_totals(
        input_tokens: u32,
        output_tokens: u32,
        cache_read_tokens: u32,
        cache_creation_tokens: u32,
        provider_cost_usd: Option<f64>,
    ) -> Self {
        Self {
            input_tokens: input_tokens
                .saturating_sub(cache_read_tokens)
                .saturating_sub(cache_creation_tokens),
            output_tokens,
            cache_read_tokens,
            cache_creation_tokens,
            provider_cost_usd,
        }
    }
}

impl Message {
    pub fn user(text: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: MessageContent::Text(text.to_string()),
        }
    }

    pub fn assistant_text(text: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: MessageContent::Text(text.to_string()),
        }
    }

    pub fn user_with_images(text: &str, images: Vec<ImageSource>) -> Self {
        let mut blocks = vec![ContentBlock::Text {
            text: text.to_string(),
        }];
        blocks.extend(
            images
                .into_iter()
                .map(|source| ContentBlock::Image { source }),
        );
        Self {
            role: "user".to_string(),
            content: MessageContent::Blocks(blocks),
        }
    }

    pub fn assistant_blocks(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: MessageContent::Blocks(blocks),
        }
    }

    pub fn tool_results(results: Vec<ContentBlock>) -> Self {
        Self {
            role: "user".to_string(),
            content: MessageContent::Blocks(results),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn user_message_has_correct_role() {
        let msg = Message::user("hello");
        assert_eq!(msg.role, "user");
    }

    #[test]
    fn user_message_serializes() {
        let msg = Message::user("hello");
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "hello");
    }

    #[test]
    fn assistant_blocks_serializes_tool_use() {
        let msg = Message::assistant_blocks(vec![
            ContentBlock::Text {
                text: "Let me check.".to_string(),
            },
            ContentBlock::ToolUse {
                id: "tu_123".to_string(),
                name: "Read".to_string(),
                input: json!({"file_path": "/tmp/test"}),
            },
        ]);
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "assistant");
        let blocks = json["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["name"], "Read");
    }

    #[test]
    fn tool_result_serializes() {
        let msg = Message::tool_results(vec![ContentBlock::ToolResult {
            tool_use_id: "tu_123".to_string(),
            content: "file contents here".to_string(),
            is_error: None,
        }]);
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "user");
        let blocks = json["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[0]["tool_use_id"], "tu_123");
        // is_error should be absent when None
        assert!(blocks[0].get("is_error").is_none());
    }

    #[test]
    fn image_message_serializes_for_anthropic() {
        let msg = Message::user_with_images(
            "describe it",
            vec![ImageSource {
                source_type: "base64".to_string(),
                media_type: "image/png".to_string(),
                data: "aGVsbG8=".to_string(),
            }],
        );
        let json = serde_json::to_value(msg).unwrap();
        assert_eq!(json["content"][1]["type"], "image");
        assert_eq!(json["content"][1]["source"]["type"], "base64");
        assert_eq!(json["content"][1]["source"]["media_type"], "image/png");
    }

    #[test]
    fn tool_result_error_serializes() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "tu_456".to_string(),
            content: "not found".to_string(),
            is_error: Some(true),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["is_error"], true);
    }

    #[test]
    fn content_block_roundtrip() {
        let original = ContentBlock::Text {
            text: "hello".to_string(),
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: ContentBlock = serde_json::from_str(&json).unwrap();
        if let ContentBlock::Text { text } = parsed {
            assert_eq!(text, "hello");
        } else {
            panic!("expected Text");
        }
    }

    #[test]
    fn reasoning_block_roundtrips_without_becoming_text() {
        let original = ContentBlock::Reasoning {
            text: Some("private thought".to_string()),
            details: vec![serde_json::json!({
                "type": "reasoning.text",
                "text": "preserve me",
                "index": 0
            })],
        };

        let json = serde_json::to_string(&original).unwrap();
        let parsed: ContentBlock = serde_json::from_str(&json).unwrap();

        assert!(matches!(
            parsed,
            ContentBlock::Reasoning { text: Some(text), details }
                if text == "private thought" && details[0]["text"] == "preserve me"
        ));
    }
}
