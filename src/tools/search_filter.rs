use std::path::{Component, Path};

pub(super) fn is_excluded(
    path: &Path,
    base: &Path,
    include_hidden: bool,
    include_build_dirs: bool,
) -> bool {
    let relative = path.strip_prefix(base).unwrap_or(path);

    relative.components().any(|component| {
        let Component::Normal(name) = component else {
            return false;
        };
        let name = name.to_string_lossy();
        (!include_hidden && name.starts_with('.')) || (!include_build_dirs && name == "target")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn excludes_hidden_and_target_components_by_default() {
        let base = Path::new("/project");

        assert!(is_excluded(
            Path::new("/project/.github/workflows/ci.yml"),
            base,
            false,
            false
        ));
        assert!(is_excluded(
            Path::new("/project/crate/target/debug/app"),
            base,
            false,
            false
        ));
        assert!(!is_excluded(
            Path::new("/project/src/targeted.rs"),
            base,
            false,
            false
        ));
    }

    #[test]
    fn flags_allow_hidden_and_target_components() {
        let base = Path::new("/project");

        assert!(!is_excluded(
            Path::new("/project/.github/workflows/ci.yml"),
            base,
            true,
            false
        ));
        assert!(!is_excluded(
            Path::new("/project/target/debug/app"),
            base,
            false,
            true
        ));
    }

    #[test]
    fn explicitly_selected_base_is_not_excluded() {
        assert!(!is_excluded(
            Path::new("/project/.github/workflows/ci.yml"),
            Path::new("/project/.github"),
            false,
            false
        ));
        assert!(!is_excluded(
            Path::new("/project/target/debug/app"),
            Path::new("/project/target"),
            false,
            false
        ));
    }
}
