#!/bin/bash
# Count lines of code for the chess engine: only .cpp, .hpp, .inl files
# Excludes external/non-project directories

cloc . \
  --exclude-dir=output,doc,script,games,saves,cutechess,stockfish,tuning,.vscode,.venv \
  --include-lang="C++,C/C++ Header" \
  --not-match-f='^ut\.hpp$' \
  --by-file-by-lang
