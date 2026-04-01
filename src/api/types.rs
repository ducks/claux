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
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: u32,
    pub cache_creation_tokens: u32,
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
}
