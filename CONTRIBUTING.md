# Contributing to RocketPy GNC

Thank you for your interest in contributing to the RocketPy Guidance, Navigation, and Control (GNC) module.

This project extends RocketPy by providing a modular GNC framework for advanced rocket simulation, state estimation, guidance algorithms, and control systems.

We welcome bug fixes, new features, documentation improvements, performance optimizations, and new GNC algorithms.

---

# Project Structure

```
rocketpy/
└── gnc/
    ├── navigation/
    ├── guidance/
    ├── control/
    ├── hardware/
    ├── mission/
    ├── utils/
    └── verification/
```

Each module should remain modular, well documented, independently testable, and focused on a single responsibility.

---

# Development Environment

## Prerequisites

Before contributing, install the following:

- Python 3.10 or later
- Git
- Docker Desktop
- Docker Compose

Docker is the recommended development environment for RocketPy GNC.

---

# Clone the Repository

```bash
git clone https://github.com/Tanaya-30/RocketPy.git
cd RocketPy
```

---

# Build the Development Container

```bash
docker compose build
```

---

# Start the Development Container

```bash
docker compose up -d
```

---

# Open a Shell Inside the Container

```bash
docker exec -it rocketgnc-dev bash
```

---

# Stop the Development Container

```bash
docker compose down
```

---

# Running Tests

Run the complete test suite:

```bash
pytest
```

Run only the GNC Navigation tests:

```bash
pytest tests/gnc/navigation -v
```

---

# Code Formatting

Format the project:

```bash
black .
```

Sort imports:

```bash
isort .
```

Run linting:

```bash
ruff check .
```

---

# Local Development (Without Docker)

Docker is recommended, but contributors may also work using a local Python virtual environment.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install RocketPy in editable mode with testing dependencies:

```bash
pip install -e ".[tests]"
```

Verify the installation:

```bash
pytest tests/gnc/navigation -v
```

---

# Coding Guidelines

Please follow these guidelines when contributing:

- Follow PEP 8.
- Write clear, readable, and well-documented code.
- Keep functions focused on a single responsibility.
- Add unit tests for all new functionality.
- Ensure all tests pass before submitting changes.
- Avoid introducing breaking changes without prior discussion.

---

# Commit Messages

Use descriptive commit messages.

Examples:

```text
feat(navigation): add quaternion integration
feat(guidance): implement ascent trajectory planner
fix(control): correct PID saturation logic
docs: update contributing guide
test(navigation): add EKF unit tests
```

---

# Pull Requests

Before opening a pull request:

- Ensure the project builds successfully.
- Run all relevant tests.
- Verify formatting and linting.
- Update documentation if necessary.
- Keep pull requests focused on a single feature or bug fix.

---

# Project Architecture

The RocketPy GNC module follows a layered architecture.

```
Navigation
     │
     ▼
Guidance
     │
     ▼
Control
     │
     ▼
Hardware Interfaces
     │
     ▼
Mission Management
```

Each layer should depend only on the layers below it. Avoid circular dependencies and design components to remain modular and reusable.

---

# Questions

If you are unsure about a design decision or implementation approach, please open an issue or start a discussion before making major architectural changes.

Thank you for contributing to RocketPy GNC!