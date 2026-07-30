.PHONY: help version-bump wait-for-ci release build test clean clippy fmt fmt-check lint install-hooks

CI_DISCOVERY_ATTEMPTS ?= 360
CI_POLL_SECONDS ?= 5

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

# Wait for the push-triggered CI run for an exact commit. This is deliberately
# separate from the tag-triggered release workflow: a release tag should not
# exist until the cross-platform test and build matrix has passed.
wait-for-ci:
	@command -v gh >/dev/null || { echo "gh is required to verify release CI"; exit 1; }
	@gh auth status >/dev/null
	@sha="$${CI_SHA:-$$(git rev-parse HEAD)}"; \
	attempt=0; \
	run_id=""; \
	echo "Waiting for CI to start for $$sha..."; \
	while [ -z "$$run_id" ] && [ "$$attempt" -lt "$(CI_DISCOVERY_ATTEMPTS)" ]; do \
		run_id="$$(gh run list --workflow ci.yml --commit "$$sha" --event push --limit 1 --json databaseId --jq '.[0].databaseId // empty')"; \
		if [ -z "$$run_id" ]; then \
			attempt=$$((attempt + 1)); \
			sleep "$(CI_POLL_SECONDS)"; \
		fi; \
	done; \
	if [ -z "$$run_id" ]; then \
		echo "Timed out waiting for CI run for $$sha"; \
		exit 1; \
	fi; \
	echo "Watching CI run $$run_id..."; \
	gh run watch "$$run_id" --exit-status

# Merge to main, push, wait for cross-platform CI, then tag and publish.
release: version-bump
	@echo "Merging into main..."
	@git checkout main
	@git merge --no-ff release/v$(VERSION) -m "Merge branch 'release/v$(VERSION)'"
	@echo "Pushing release commit to origin..."
	@git push origin main
	@$(MAKE) wait-for-ci CI_SHA=$$(git rev-parse HEAD)
	@echo "Creating tag v$(VERSION) after CI passed..."
	@git tag -a v$(VERSION) -m "Release v$(VERSION)"
	@git push origin v$(VERSION)
	@echo "Publishing to crates.io..."
	@cargo publish
	@echo ""
	@echo "Released v$(VERSION)"
	@echo "  - Merged release/v$(VERSION) into main"
	@echo "  - Passed cross-platform CI"
	@echo "  - Tagged and pushed v$(VERSION)"
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

# Run rustfmt to format the code
fmt:
	cargo fmt

# Check that rustfmt is satisfied without modifying files (mirrors CI)
fmt-check:
	cargo fmt -- --check

# Run all the checks CI runs, in order. Cheap to run locally before pushing.
lint: fmt-check
	cargo clippy -- -D warnings
	cargo test

# Install a pre-push hook that runs `make lint` before any push, so CI
# failures from formatting / clippy / tests are caught locally.
install-hooks:
	@mkdir -p .git/hooks
	@printf '#!/usr/bin/env bash\nset -e\nexec make lint\n' > .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "Installed pre-push hook -> make lint"
