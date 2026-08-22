use crate::api::types::ImageSource;
use anyhow::{bail, Context, Result};
use base64::Engine;
use std::path::{Path, PathBuf};

const MAX_IMAGE_BYTES: u64 = 20 * 1024 * 1024;

pub fn load_images(paths: &[PathBuf]) -> Result<Vec<ImageSource>> {
    paths.iter().map(|path| load_image(path)).collect()
}

pub fn load_image(path: &Path) -> Result<ImageSource> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("failed to inspect image {}", path.display()))?;
    if !metadata.is_file() {
        bail!("image is not a regular file: {}", path.display());
    }
    if metadata.len() > MAX_IMAGE_BYTES {
        bail!(
            "image exceeds the 20 MiB attachment limit: {}",
            path.display()
        );
    }

    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read image {}", path.display()))?;
    let media_type = detect_media_type(&bytes).ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported image format for {}; expected PNG, JPEG, GIF, or WebP",
            path.display()
        )
    })?;

    Ok(ImageSource {
        source_type: "base64".to_string(),
        media_type: media_type.to_string(),
        data: base64::engine::general_purpose::STANDARD.encode(bytes),
    })
}

fn detect_media_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_supported_image_signatures() {
        assert_eq!(
            detect_media_type(b"\x89PNG\r\n\x1a\nrest"),
            Some("image/png")
        );
        assert_eq!(detect_media_type(b"\xff\xd8\xffrest"), Some("image/jpeg"));
        assert_eq!(detect_media_type(b"GIF89arest"), Some("image/gif"));
        assert_eq!(detect_media_type(b"RIFFxxxxWEBPrest"), Some("image/webp"));
        assert_eq!(detect_media_type(b"not an image"), None);
    }

    #[test]
    fn loads_and_encodes_an_image() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.png");
        std::fs::write(&path, b"\x89PNG\r\n\x1a\ncontents").unwrap();
        let image = load_image(&path).unwrap();
        assert_eq!(image.media_type, "image/png");
        assert_eq!(image.source_type, "base64");
        assert!(!image.data.is_empty());
    }
}
