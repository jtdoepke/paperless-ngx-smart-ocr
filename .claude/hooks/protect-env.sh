#!/bin/bash
# protect-env.sh - Block reads to .env and .envrc files
# This hook runs before Read tool calls and denies access to environment files

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Check if file matches .env or .envrc patterns
if [[ "$FILE_PATH" =~ (^|/)\.env($|\..*$) ]] || [[ "$FILE_PATH" =~ (^|/)\.envrc$ ]]; then
    # Output decision to block
    jq -n \
        --arg reason "Access denied: Reading .env and .envrc files is not allowed to protect secrets" \
        '{
            hookSpecificOutput: {
                hookEventName: "PreToolUse",
                permissionDecision: "deny",
                permissionDecisionReason: $reason
            }
        }'
    exit 0
fi

# Allow the tool call (no output needed)
exit 0
