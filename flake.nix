{
  description = "Python ML Flake";

  nixConfig = {
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachSystem ["x86_64-linux"] (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = false;
          };
        };
      in {
        formatter = pkgs.alejandra;
        devShells.default =
          (pkgs.buildFHSUserEnv {
            name = "machine-learning";
            targetPkgs = pkgs:
              with pkgs; [
                python311
                python311Packages.pip
                python311Packages.virtualenv
                python311Packages.python-lsp-server
                autoAddDriverRunpath
                cudaPackages.cudatoolkit
              ];
            runScript = "bash";
          })
          .env;
      }
    );
}
