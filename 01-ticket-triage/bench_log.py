import csv, os

def log_result(row, path="results.csv"):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:          # "a" = append, never overwrite
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            writer.writeheader()                     # column names, first time only
        writer.writerow(row)