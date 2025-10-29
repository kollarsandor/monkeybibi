{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    julia-bin
    python311Full
    python311Packages.pip
    python311Packages.numpy
    python311Packages.pytorch
    coq
    agda
    lean
    isabelle
    ocaml
    ocamlPackages.findlib
    swi-prolog
  ];

  shellHook = ''
    echo "🚀 AlphaFold3 Multi-Language Environment"
    echo "✅ Julia installed"
    echo "✅ Python 3.11 installed"
    echo "✅ Coq, Agda, Lean, Isabelle installed"
    echo "✅ OCaml, Prolog installed"
  '';
}
