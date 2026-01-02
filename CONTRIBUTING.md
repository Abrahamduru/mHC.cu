# Contributing

## Pre-commit Hook

This project uses a pre-commit hook to automatically format code using the rules in `.clang-format` and run tests before each commit.

**Setup:**
```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

**Process:**
1. Runs `clang-format` on all staged `.cu` and `.cuh` files
2. Builds the project
3. Runs all tests

If any step fails, the commit will be aborted.

**Format Manually:**
```bash
find csrc -name "*.cu" -o -name "*.cuh" | xargs clang-format -i
```

## Code Style

- 4 space indents
- 100 column limit
- K & R braces
- Left pointer alignment
