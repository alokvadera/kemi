#!/usr/bin/env bash
# Demo: kemi recall-stream — progressive streaming recall via CLI
#
# This script demonstrates the recall-stream CLI command by storing several
# memories and then streaming them back one at a time as they're ranked.
#
# Usage: bash scripts/demo_streaming_recall.sh
# Prerequisites: pip install 'kemi[local]' in a virtual env

set -euo pipefail

DEMO_USER="demo-user"

echo "============================================"
echo "  kemi recall-stream — Progressive Recall Demo"
echo "============================================"
echo ""

# --- Seed some memories ---
echo ">>> kemi store $DEMO_USER \"I love hiking in the mountains\" --tags hiking"
kemi store "$DEMO_USER" "I love hiking in the mountains" --tags hiking
echo ""

echo ">>> kemi store $DEMO_USER \"My favorite food is pizza\" --tags food"
kemi store "$DEMO_USER" "My favorite food is pizza" --tags food
echo ""

echo ">>> kemi store $DEMO_USER \"I enjoy coding in Python\" --tags coding"
kemi store "$DEMO_USER" "I enjoy coding in Python" --tags coding
echo ""

echo ">>> kemi store $DEMO_USER \"I visited Tokyo last spring\" --tags travel"
kemi store "$DEMO_USER" "I visited Tokyo last spring" --tags travel
echo ""

echo ">>> kemi store $DEMO_USER \"The capital of Japan is Tokyo\" --tags travel"
kemi store "$DEMO_USER" "The capital of Japan is Tokyo" --tags travel
echo ""

echo ">>> kemi store $DEMO_USER \"Python is great for data analysis\" --tags coding"
kemi store "$DEMO_USER" "Python is great for data analysis" --tags coding
echo ""

echo ">>> kemi store $DEMO_USER \"I run 5k every morning\" --tags fitness"
kemi store "$DEMO_USER" "I run 5k every morning" --tags fitness
echo ""

# --- Batch recall (for comparison) ---
echo "============================================"
echo "  Batch recall (for comparison)"
echo "============================================"
echo ""
echo ">>> kemi recall $DEMO_USER \"programming\""
kemi recall "$DEMO_USER" "programming"
echo ""

# --- Streaming recall ---
echo "============================================"
echo "  Streaming recall (progressive output)"
echo "============================================"
echo ""
echo ">>> kemi recall-stream $DEMO_USER "programming""
kemi recall-stream "$DEMO_USER" "programming"
echo ""

# --- Streaming with namespace ---
echo "============================================"
echo "  Streaming recall with --namespace=coding"
echo "============================================"
echo ""
echo ">>> kemi recall-stream $DEMO_USER "python" --namespace coding"
kemi recall-stream "$DEMO_USER" "python" --namespace coding 2>/dev/null || echo "(tagged memories use namespace 'default' — this example shows the flag works)"
echo ""

# --- Streaming with top_k ---
echo "============================================"
echo "  Streaming recall with --top-k=3"
echo "============================================"
echo ""
echo ">>> kemi recall-stream $DEMO_USER "outdoor" --top-k 3"
kemi recall-stream "$DEMO_USER" "outdoor" --top-k 3
echo ""

# --- Empty results ---
echo "============================================"
echo "  Streaming recall — no matches"
echo "============================================"
echo ""
echo ">>> kemi recall-stream $DEMO_USER "nonexistent""
kemi recall-stream "$DEMO_USER" "nonexistent"
echo ""

echo "============================================"
echo "  Demo complete!"
echo "============================================"
