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

3a. **Install development dependencies (optional)**
```bash
pip install -e ".[dev]"
```

4. **Run the simulation**
```bash
python main.py
```

### Linting (requires development dependencies)

```bash
# Check all files in the project
ruff check .

# Optional: automatically fix some issues
ruff check . --fix

# Run mypy to perform type checking on Python files
mypy .
```