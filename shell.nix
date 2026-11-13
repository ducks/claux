{ pkgs ? import <nixpkgs> {} }:

let
  rust-overlay = import (builtins.fetchTarball
    "https://github.com/oxalica/rust-overlay/archive/master.tar.gz");
  pkgs' = import <nixpkgs> { overlays = [ rust-overlay ]; };
  rust = pkgs'.rust-bin.stable.latest.default;
in
pkgs'.mkShell {
  buildInputs = with pkgs'; [
    rust
    rust-analyzer
    pkg-config
    openssl
    python3
  ];

  RUST_BACKTRACE = 1;
}
