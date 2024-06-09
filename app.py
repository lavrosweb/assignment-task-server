from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time

app = Flask(__name__)
CORS(app)

def record_step(matrix, step_description, steps):
    # Convert all numbers to strings for consistency in the output
    state = {
        "description": step_description,
        "matrix": [[str(int(item)) for item in row] for row in matrix]
    }
    steps.append(state)

def reduce_matrix(matrix, steps):
    n = len(matrix)

    # Step 1: Subtract the smallest value in each row from all entries in that row.
    row_min = np.min(matrix, axis=1)
    matrix -= row_min[:, np.newaxis]
    record_step(matrix, "Вычитание минимума у строк", steps)

    # Step 2: Subtract the smallest value in each column from all entries in that column.
    col_min = np.min(matrix, axis=0)
    matrix -= col_min
    record_step(matrix, "Вычитание минимума у столбцов", steps)

    return matrix

def cover_zeros(matrix, steps):
    n = len(matrix)
    zero_mask = (matrix == 0)
    covered_rows = np.zeros(n, dtype=bool)
    covered_cols = np.zeros(n, dtype=bool)
    row_coverage = np.zeros(n, dtype=bool)
    col_coverage = np.zeros(n, dtype=bool)

    while True:
        row_coverage = np.any(zero_mask & ~col_coverage, axis=1)
        col_coverage = np.any(zero_mask & ~row_coverage[:, np.newaxis], axis=0)
        
        if not np.any(zero_mask & ~row_coverage[:, np.newaxis] & ~col_coverage):
            break

    record_step(matrix, "Закрываем нули минимальным количеством строк", steps)
    return row_coverage, col_coverage

def adjust_matrix(matrix, covered_rows, covered_cols, steps):
    uncovered_values = matrix[~covered_rows[:, np.newaxis] & ~covered_cols]
    if uncovered_values.size > 0:
        min_uncovered = uncovered_values.min()
    else:
        min_uncovered = 0

    matrix[~covered_rows[:, np.newaxis] & ~covered_cols] -= min_uncovered
    matrix[covered_rows[:, np.newaxis] & covered_cols] += min_uncovered

    record_step(matrix, f"Adjust matrix by subtracting {int(min_uncovered)} and adding to intersections", steps)
    return min_uncovered

def hungarian_algorithm(cost_matrix):
    steps = []

    # Step: Матрица первоначальных затрат
    initial_matrix = cost_matrix.copy()
    record_step(initial_matrix, "Матрица первоначальных затрат", steps)

    matrix = cost_matrix.copy()

    matrix = reduce_matrix(matrix, steps)

    while True:
        row_coverage, col_coverage = cover_zeros(matrix, steps)
        total_coverage = np.sum(row_coverage) + np.sum(col_coverage)

        if total_coverage >= len(matrix):
            break

        adjust_matrix(matrix, row_coverage, col_coverage, steps)

    return steps, matrix

def solve_assignment_problem(cost_matrix):
    steps = []

    start_time = time.time()

    # Execute the Hungarian algorithm
    step_results, final_matrix = hungarian_algorithm(cost_matrix)
    steps.extend(step_results)

    n = len(final_matrix)
    result_matrix = np.zeros_like(cost_matrix, dtype=int)
    for i in range(n):
        for j in range(n):
            if final_matrix[i, j] == 0:
                result_matrix[i, j] = 1

    end_time = time.time()
    execution_time = end_time - start_time

    optimal_plan = [(i, j) for i in range(n) for j in range(n) if result_matrix[i, j] == 1]
    total_cost = sum(cost_matrix[i, j] for i, j in optimal_plan)

    optimal_plan_str = [[str(i), str(j)] for i, j in optimal_plan]
    total_cost_str = str(int(total_cost))
    execution_time_str = str(execution_time)
    steps.append({"description": "Финальная матрица", "matrix": [[str(int(item)) for item in row] for row in result_matrix]})

    return {
        "steps": steps,
        "optimal_plan": optimal_plan_str,
        "total_cost": total_cost_str,
        "execution_time": execution_time_str
    }

@app.route('/solve_assignment', methods=['POST'])
def solve_assignment():
    data = request.json
    if 'cost_matrix' not in data:
        return jsonify({"error": "Missing 'cost_matrix' key in JSON request"}), 400
    
    try:
        cost_matrix = np.array(data['cost_matrix'], dtype=np.int64)
    except ValueError:
        return jsonify({"error": "Invalid 'cost_matrix' format. Ensure all elements are numbers."}), 400

    if cost_matrix.ndim != 2 or cost_matrix.shape[0] != cost_matrix.shape[1]:
        return jsonify({"error": "'cost_matrix' must be a square matrix"}), 400

    result = solve_assignment_problem(cost_matrix)

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)