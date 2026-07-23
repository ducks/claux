/// Basic HTML tag stripping. Not a full parser — just removes tags and
/// decodes common entities to make HTML content readable.
pub fn strip_html(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let mut last_was_whitespace = false;

    let lower = html.to_lowercase();
    let chars: Vec<char> = html.chars().collect();
    let lower_chars: Vec<char> = lower.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        if !in_tag && chars[i] == '<' {
            let remaining: String = lower_chars[i..].iter().take(20).collect();
            if remaining.starts_with("<script") {
                in_script = true;
            } else if remaining.starts_with("</script") {
                in_script = false;
            } else if remaining.starts_with("<style") {
                in_style = true;
            } else if remaining.starts_with("</style") {
                in_style = false;
            }
            in_tag = true;
            i += 1;
            continue;
        }

        if in_tag {
            if chars[i] == '>' {
                in_tag = false;
            }
            i += 1;
            continue;
        }

        if in_script || in_style {
            i += 1;
            continue;
        }

        if chars[i] == '&' {
            let remaining: String = chars[i..].iter().take(10).collect();
            let decoded = [
                ("&amp;", '&'),
                ("&lt;", '<'),
                ("&gt;", '>'),
                ("&quot;", '"'),
                ("&#39;", '\''),
                ("&apos;", '\''),
                ("&nbsp;", ' '),
            ]
            .into_iter()
            .find(|(entity, _)| remaining.starts_with(entity));
            if let Some((entity, character)) = decoded {
                result.push(character);
                i += entity.chars().count();
                last_was_whitespace = character.is_whitespace();
                continue;
            }
        }

        if chars[i].is_whitespace() {
            if !last_was_whitespace {
                result.push(' ');
                last_was_whitespace = true;
            }
        } else {
            result.push(chars[i]);
            last_was_whitespace = false;
        }

        i += 1;
    }

    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_basic_tags() {
        assert_eq!(strip_html("<p>Hello <b>world</b></p>"), "Hello world");
    }

    #[test]
    fn decodes_entities() {
        assert_eq!(strip_html("&amp; &lt; &gt; &quot;"), "& < > \"");
    }

    #[test]
    fn removes_script_and_style_content() {
        let html = "before<script>bad()</script><style>.bad{}</style>after";
        assert_eq!(strip_html(html), "beforeafter");
    }

    #[test]
    fn collapses_whitespace() {
        assert_eq!(strip_html("hello    \n\n   world"), "hello world");
    }
}
