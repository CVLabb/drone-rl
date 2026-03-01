# DroneRL

DroneRL is a Python backend project focused on building a basic
rigid-body physics simulation for a simplified quadrotor-style drone.

The current scope is limited to modeling physical dynamics and thruster
forces. Reinforcement learning and higher-level control logic are
planned but not yet implemented.

------------------------------------------------------------------------

## Project Structure

```
drone-rl/
└─ backend/                  # Python backend (physics simulation)
   ├─ core/                  # Core simulation and business logic
   ├─ tests/                 # Unit tests
   │  └─ core_tests/               # Mirrors core modules
   ├─ main.py                # Program entry point and simulation runner
   └─ pyproject.toml         # Backend dependencies and configuration
```

------------------------------------------------------------------------

## Backend Setup

### Requirements

-   Python 3.10+

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
```

2. **Create and activate a virtual environment**
```bash
cd drone-rl/backend
```
- Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

- macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install backend dependencies:**
```bash
pip install -e .
```

4. **Run the simulation**
```bash
python main.py
```

------------------------------------------------------------------------

## Development Environment (Tooling & Quality Gates) 

### Installation 

1. **Install backend development dependencies:** 

From inside the activated backend virtual environment:

```bash
pip install -e ".[dev]"
```

This installs: 
- Ruff (linting) 
- Ruff formatter 
- Mypy (type checking) 
- Pytest (tests) 

2. **Install pre-commit and associated git hook** 

From the repository root:

```bash
pipx install pre-commit
pre-commit install
```

3. **Run pre-commit manually to validate**

```bash
pre-commit run --all-files
```

Pre-commit automatically runs on every commit:
- Ruff (lint)
- Ruff-format (formatting)
- Mypy (type checking)
- Pytest (backend tests)

### Manual Quality Commands (Optional)

```bash
# Check all files in the project
ruff check .

# Optional: automatically fix some issues
ruff check . --fix

# Run Ruff's formatter explicitly
ruff format .

# Run mypy to perform type checking on Python files
mypy .

# Run all tests
pytest
```

------------------------------------------------------------------------