{
  description = "narratelm — narrate epub books into audiobooks locally with VibeVoice";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            pkgs.ffmpeg
            pkgs.just
            pkgs.git
            pkgs.git-lfs
          ];

          shellHook = ''
            # Keep uv on the nix-provided interpreter instead of downloading one.
            export UV_PYTHON="${python}/bin/python3"

            ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              # manylinux wheels (numpy, lxml, soundfile) need libstdc++ & friends,
              # which NixOS does not put on the default loader path.
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            ''}

            ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              # The repo may be shared with a macOS host (VM setup): keep the
              # Linux venv separate so the two platforms don't clobber each other.
              export UV_PROJECT_ENVIRONMENT="$PWD/.venv-linux-$(uname -m)"
            ''}
            VENV_DIR="''${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"

            # Put the project venv's entry points (narratelm, pytest, ...) on PATH.
            # Harmless before `just setup` creates the venv.
            export PATH="$VENV_DIR/bin:$PATH"

            if [ ! -d "$VENV_DIR" ]; then
              echo "narratelm: no venv yet — run 'just setup' to install dependencies"
            fi
          '';
        };
      });
}
