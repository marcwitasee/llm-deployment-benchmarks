# The one prompt every model sees, in training AND in testing. Keeping it in one file means the
# training data (make_positions.py) and the evaluation (eval_positions.py) can never drift apart.

PROMPT = (
    "You are playing chess. Here is the current position in FEN notation:\n"
    "{fen}\n"
    "Reply with only the best move in UCI notation (for example e2e4 or g1f3)."
)

# Experiment 2: same prompt plus every legal move, so the model only has to choose, not work out the rules.
PROMPT_WITH_LEGAL_MOVES = (
    "You are playing chess. Here is the current position in FEN notation:\n"
    "{fen}\n"
    "The legal moves are: {legal_moves}\n"
    "Reply with only the best move from that list, in UCI notation (for example e2e4 or g1f3)."
)


def legal_moves_text(fen):
    import chess
    return " ".join(sorted(m.uci() for m in chess.Board(fen).legal_moves))
