# The one prompt every model sees, in training AND in testing. Keeping it in one file means the
# training data (make_positions.py) and the evaluation (eval_positions.py) can never drift apart.

PROMPT = (
    "You are playing chess. Here is the current position in FEN notation:\n"
    "{fen}\n"
    "Reply with only the best move in UCI notation (for example e2e4 or g1f3)."
)
