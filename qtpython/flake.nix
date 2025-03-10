
{
  description = "Qt6 + Python Dev Environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";

  outputs = { self, nixpkgs }: 
    let 
      system = "x86_64-linux"; 
      pkgs = nixpkgs.legacyPackages.${system}; 
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          pkgs.python3
          pkgs.python3Packages.pyqtgraph
          pkgs.python3Packages.pyqt6
          pkgs.qt6.qtbase
          pkgs.qt6.wrapQtAppsHook
        ];

        shellHook = ''
          export QT_QPA_PLATFORM_PLUGIN_PATH=${pkgs.qt6.qtbase}/lib/qt-*/plugins
        '';
      };
    };
}
