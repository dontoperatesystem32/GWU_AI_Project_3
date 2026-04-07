# Generalized Tic Tac Toe AI

This repository contains a generalized Tic Tac Toe agent for an `n x n` board with a configurable win condition `m`. It includes:

- a local AI-vs-AI simulator in `main.py`
- an online game client in `api.py` for the NotExponential AI P2P gaming API
- unit tests in `test_main.py`

## Project Files

- `main.py` - board model, evaluation logic, move ordering, minimax agent, and local game runner
- `api.py` - API client and automated runner for online matches
- `test_main.py` - unit tests for board state handling and agent behavior

## How the Code Works

### 1. Board Representation

The `Board` class in `main.py` stores:

- the grid state
- move count
- winner state
- a Zobrist hash for fast state lookup
- precomputed windows of length `m` for rows, columns, and diagonals
- frontier cells near existing moves to reduce the branching factor

Instead of scanning the full board from scratch after every move, the code updates cached line counts and heuristic scores incrementally. That makes search much faster on larger boards.

### 2. Evaluation Function

The evaluation combines two ideas:

- line potential: open sequences of `X` or `O` inside windows of length `m`
- center control: earlier moves near the center are rewarded more strongly

Mixed windows containing both players do not contribute to the score. Stronger threats such as `m-1` in a row are weighted much more heavily.

### 3. Search Strategy

The `MiniMaxAgent` uses:

- negamax with alpha-beta pruning
- iterative deepening
- move ordering based on immediate wins, forced blocks, forks, defense, and center bias
- a transposition table keyed by Zobrist hash
- a time budget per move

This lets the agent return a legal move quickly even with a short time limit, while searching deeper when more time is available.

### 4. Local Runner

The `Game` class in `main.py` creates two AI agents and lets them play each other until there is a win or draw.

### 5. API Runner

The code in `api.py` connects the AI to the NotExponential platform. It can:

- create or reuse a team
- create or join a game
- poll the server for the latest board state
- compute a move locally
- submit the move back to the server

Credentials can be provided through environment variables:

- `TTT_USER_ID`
- `TTT_API_KEY`

The API runner also loads these values from a local `.env` file. The `.env`
file is ignored by Git.

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API credentials

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Then edit `.env` with your own NotExponential credentials:

```text
TTT_USER_ID=your_user_id
TTT_API_KEY=your_api_key
```

## Running Tutorial

### Run the local AI vs AI version

Start the simulator:

```bash
python3 main.py
```

You will be prompted for:

- board size `n`
- win length `m`
- time limit in seconds
- max search depth

Example:

```text
Board size n: 5
Win length m: 4
Time limit (seconds): 1.5
Max depth: 5
```

The program will then print the board after each move and finish with either a winner or a draw.

### Run the online API agent

Make sure `.env` contains your `TTT_USER_ID` and `TTT_API_KEY`, or export those
variables in your shell before running the script.

Start the API runner:

```bash
python3 api.py
```

The script will walk you through:

1. choosing an existing team or creating a new one
2. optionally adding a teammate
3. creating a new game or entering an existing `gameId`
4. choosing the search time limit and max depth

After setup, the program will keep polling the game and automatically submit moves when it is your turn.

### Run the tests

```bash
python3 -m unittest test_main.py test_api.py
```

## Notes

- `main.py` runs a local simulation and does not require network access.
- `api.py` requires internet access and the external game service to be available.
- The tests use Python's built-in `unittest` module, so no extra test framework is required.
