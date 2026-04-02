.PHONY: help version-bump release build test clean clippy

# Auto-generate version from today's date with auto-incrementing patch
# Format: YYYYMMDD.0.X where X increments if releasing multiple times per day
define get_next_version
$(shell \
	TODAY=$$(date +%Y%m%d); \
	LATEST=$$(git tag -l "v$$TODAY.*" 2>/dev/null | sort -V | tail -1); \
	if [ -z "$$LATEST" ]; then \
		echo "$$TODAY.0.0"; \
	else \
		PATCH=$$(echo "$$LATEST" | sed 's/.*\.0\.\([0-9]*\)/\1/'); \
		echo "$$TODAY.0.$$((PATCH + 1))"; \
	fi \
)
endef

VERSION := $(get_next_version)

help:
	@echo "claux Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make release                       - Auto-version and release (recommended)"
	@echo "  make release VERSION=20260125.0.0  - Release with specific version"
	@echo "  make build                         - Build release binary"
	@echo "  make test                          - Run tests"
	@echo "  make clippy                        - Run clippy"
	@echo "  make clean                         - Clean build artifacts"
	@echo ""
	@echo "Next version will be: $(VERSION)"

# Bump version in Cargo.toml and commit on a branch
version-bump:
	@echo "Next version: $(VERSION)"
	@echo "Creating release branch for version $(VERSION)..."
	@git checkout -b release/v$(VERSION)
	@echo "Bumping version to $(VERSION)..."
	@sed -i 's/^version = .*/version = "$(VERSION)"/' Cargo.toml
	@echo "Updating Cargo.lock..."
	@cargo check --quiet 2>/dev/null || true
	@git add Cargo.toml Cargo.lock
	@git commit -m "chore: bump version to $(VERSION)"
	@echo ""
	@echo "Created branch release/v$(VERSION)"
	@echo "Version bumped to $(VERSION)"
	@echo "Commit created"

# Merge to main, tag, push, and publish to crates.io
release: version-bump
	@echo "Merging into main..."
	@git checkout main
	@git merge --no-ff release/v$(VERSION) -m "Merge branch 'release/v$(VERSION)'"
	@echo "Creating tag v$(VERSION) on main..."
	@git tag -a v$(VERSION) -m "Release v$(VERSION)"
	@echo "Pushing to origin..."
	@git push origin main
	@git push origin v$(VERSION)
	@echo "Publishing to crates.io..."
	@cargo publish
	@echo ""
	@echo "Released v$(VERSION)"
	@echo "  - Merged release/v$(VERSION) into main"
	@echo "  - Tagged v$(VERSION)"
	@echo "  - Pushed to GitHub"
	@echo "  - Published to crates.io"

# Build release binary
build:
	cargo build --release

# Run tests
test:
	cargo test

# Run clippy
clippy:
	cargo clippy -- -D warnings

# Clean build artifacts
clean:
	cargo clean
