#!/bin/bash
# WSL2 の NVIDIA ドライバはコンテナに libcuda.so.1 (バージョン付き) だけをマウントし、
# 無印の libcuda.so シンボリックリンクを作らない。Genesis の内部コンパイラ (quadrants)
# は無印の libcuda.so を dlopen しようとして失敗するので、起動時に自分で作る
# (ビルド時にはまだマウントされていないので Dockerfile 側では作れない)。
# 2026-10-16, RTX 5090 / WSL2 で確認。
set -e
lib=$(find / -maxdepth 6 -name 'libcuda.so.1' 2>/dev/null | head -1)
if [ -n "$lib" ]; then
  dir=$(dirname "$lib")
  [ -e "$dir/libcuda.so" ] || ln -sf "$lib" "$dir/libcuda.so"
  ldconfig 2>/dev/null || true
fi
exec "$@"
