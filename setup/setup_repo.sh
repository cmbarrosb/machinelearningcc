#!/usr/bin/env bash
set -e

# ---- Configuration (no placeholders) ----
GITHUB_USER="cmbarrosb"
REPO_NAME="machinelearningcc"
VISIBILITY="public"
CLONE_DIR="$HOME/machinelearningcc"
LOG_FILE="$CLONE_DIR/output.txt"

# ---- Prep parent directory ----
mkdir -p "$(dirname "$CLONE_DIR")"

# ---- Ensure clone directory exists for logging ----
mkdir -p "$CLONE_DIR"
# ---- Start logging ----
exec > >(tee "$LOG_FILE") 2>&1

echo "===== $(date) ====="
echo "1. Creating GitHub repository ${GITHUB_USER}/${REPO_NAME}..."
gh repo create "${GITHUB_USER}/${REPO_NAME}" --"${VISIBILITY}" --confirm

echo "2. Cloning into ${CLONE_DIR}..."
gh repo clone "${GITHUB_USER}/${REPO_NAME}" "${CLONE_DIR}"

cd "${CLONE_DIR}"

echo "3. Creating README.md..."
cat << 'EOF' > README.md
# machinelearningcc

Repository for Machine Learning project on distributed MNIST training.
EOF

echo "4. Committing and pushing README..."
git add README.md
git commit -m "Add initial README"
git push origin main

echo "Setup complete."
echo "All output logged to ${LOG_FILE}"
