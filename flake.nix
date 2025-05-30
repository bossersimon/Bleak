
{
  description = "Qt6 + Python Dev Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/f0946fa5f1fb876a9dc2e1850d9d3a4e3f914092";
    flakeutils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flakeutils }: 
    flakeutils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system};

      python = pkgs.python312;

      in

      # Some python packages have dependencies that 
      # are broken on 32-bit systems. Hence, 
      # we have this if case here. We have no results
      # in this flake for such systems. 
      if !(pkgs.lib.hasInfix "i686" system) then {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            (python.withPackages (p: [
              p.pyqtgraph
              p.pyqt6
              p.qasync

              p.scipy
              p.bleak
            ]))
            pkgs.qt6.qtbase
#            pkgs.qt6.wrapQtAppsHook
            pkgs.gtk-engine-murrine
          ];
          shellHook = ''
                export QT_PLUGIN_PATH=${pkgs.qt6.qtbase}/lib/qt-*/plugins
                export QT_QPA_PLATFORM_PLUGIN_PATH=${pkgs.qt6.qtbase}/lib/qt-*/plugins
          '';
        };

      } else {}
    );
}
