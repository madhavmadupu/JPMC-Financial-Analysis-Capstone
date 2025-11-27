import pandas as pd
import numpy as np

def load_loan_data(file_path):
    """Load and parse the loan data from the CSV file."""
    # Read the CSV file. Since it's a single row with many columns, we need to transpose it.
    df = pd.read_csv(file_path, header=None)
    # Transpose to get one row per record
    df = df.T
    # Assign column names based on the description in the task
    # The last column (index 6) is the FICO score
    df.columns = ['loan_id', 'default_flag', 'monthly_income', 'total_debt', 'total_assets', 'loan_amount', 'fico_score']
    return df

def create_buckets_dp(fico_scores, defaults, num_buckets=10):
    """
    Use dynamic programming to find the optimal bucket boundaries for FICO scores
    that maximize the log-likelihood function.

    Args:
        fico_scores: Array of FICO scores.
        defaults: Array of default flags (0 or 1).
        num_buckets: Number of buckets to create.

    Returns:
        List of bucket boundaries.
    """
    # Sort the data by FICO score to ensure monotonicity
    sorted_indices = np.argsort(fico_scores)
    sorted_fico = fico_scores[sorted_indices]
    sorted_defaults = defaults[sorted_indices]

    n = len(sorted_fico)

    # DP table: dp[i][j] will store the maximum log-likelihood for partitioning
    # the first i scores into j buckets.
    # We'll also store the boundaries to reconstruct the solution.
    dp = np.full((n + 1, num_buckets + 1), -np.inf)
    # Initialize base case: 0 buckets for 0 scores has log-likelihood 0.
    dp[0][0] = 0.0

    # parent[i][j] will store the best split point for the last bucket when
    # partitioning the first i scores into j buckets.
    parent = np.zeros((n + 1, num_buckets + 1), dtype=int)

    # Fill the DP table
    for i in range(1, n + 1):  # For each number of scores considered
        for j in range(1, min(i, num_buckets) + 1):  # For each number of buckets
            # Try all possible positions for the last bucket boundary
            for k in range(j - 1, i):  # k is the start index of the last bucket (inclusive)
                # Calculate the log-likelihood for the last bucket (from k to i-1)
                bucket_defaults = sorted_defaults[k:i]
                bucket_size = len(bucket_defaults)
                num_defaults = np.sum(bucket_defaults)
                if bucket_size == 0:
                    continue
                p = num_defaults / bucket_size  # Probability of default in this bucket
                if p == 0 or p == 1:
                    # Avoid log(0) or log(1) issues; use a small epsilon if needed for numerical stability
                    ll_bucket = 0.0
                else:
                    ll_bucket = num_defaults * np.log(p) + (bucket_size - num_defaults) * np.log(1 - p)

                # The total log-likelihood is the sum of the previous state and this bucket
                if dp[k][j - 1] != -np.inf:
                    total_ll = dp[k][j - 1] + ll_bucket
                    if total_ll > dp[i][j]:
                        dp[i][j] = total_ll
                        parent[i][j] = k

    # Reconstruct the bucket boundaries
    boundaries = []
    current_i = n
    current_j = num_buckets
    while current_j > 0:
        start_idx = parent[current_i][current_j]
        # The boundary is the FICO score at the start of the bucket
        if start_idx < n:  # Ensure we don't go out of bounds
            boundaries.append(sorted_fico[start_idx])
        current_i = start_idx
        current_j -= 1

    # Add the lower bound (min FICO score) and sort
    boundaries.append(sorted_fico[0])  # Lower bound
    boundaries.sort()

    # Add the upper bound (max FICO score + 1) to complete the range
    boundaries.append(sorted_fico[-1] + 1)

    return boundaries

# Main execution
if __name__ == "__main__":
    # Load the data
    file_path = "Task 3 and 4_Loan_Data (1).csv"
    loan_data = load_loan_data(file_path)

    # Extract FICO scores and default flags
    fico_scores = loan_data['fico_score'].values
    defaults = loan_data['default_flag'].values

    # Define the number of buckets (you can adjust this based on model requirements)
    num_buckets = 10

    # Create the optimal buckets
    optimal_boundaries = create_buckets_dp(fico_scores, defaults, num_buckets)

    print("Optimal Bucket Boundaries:")
    print(optimal_boundaries)

    # Optionally, you can assign each loan to a bucket and calculate the average default rate per bucket
    # This is useful for the risk manager to analyze the data.
    loan_data['bucket'] = pd.cut(loan_data['fico_score'], bins=optimal_boundaries, labels=False, right=False)
    bucket_stats = loan_data.groupby('bucket').agg(
        count=('default_flag', 'count'),
        defaults=('default_flag', 'sum'),
        avg_default_rate=('default_flag', 'mean')
    ).reset_index()

    print("\nBucket Statistics:")
    print(bucket_stats)