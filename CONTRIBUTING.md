# Contributing to RealignR

Thank you for your interest in contributing to the RealignR optimizer project! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine:
   ```bash
   git clone https://github.com/yourusername/realignr.git
   cd realignr
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Create a branch** for your feature or bugfix:
   ```bash
   git checkout -b feature-name
   ```

## Development Guidelines

### Code Style

We follow PEP 8 style guidelines for Python code:
- Use 4 spaces for indentation
- Maximum line length of 100 characters
- Add docstrings to all functions and classes

### Testing

- Add tests for new features
- Ensure all existing tests pass before submitting your contribution
- Run tests with:
  ```bash
  python -m unittest discover tests
  ```

### Optimizer Implementation

When modifying the core RealignR optimizer:
1. Maintain backward compatibility with the original API when possible
2. Document the mathematical reasoning behind your changes
3. Include benchmark results comparing to the original version

### Leaderboard and Dashboard System

When enhancing the leaderboard or dashboard functionality:
1. Maintain compatibility with existing result file formats
2. Add clear documentation for any new metrics or visualizations
3. Test with a variety of result formats and dataset configurations

## Submitting Contributions

1. **Update documentation** for your changes
2. **Add or update tests** as needed
3. **Run all tests** to make sure they pass
4. **Commit your changes**:
   ```bash
   git commit -am "Add feature: description of changes"
   ```
5. **Push to your fork**:
   ```bash
   git push origin feature-name
   ```
6. **Create a pull request** from your forked repository to the original repository

## Reporting Issues

When reporting issues, please include:
- A clear description of the problem
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- System information (OS, Python version, dependency versions)
- Any relevant error messages or logs

## Feature Requests

Feature requests are welcome! Please include:
- A clear description of the feature
- The motivation for the feature
- Example use cases
- Potential implementation approach, if known

Thank you for contributing to RealignR!
