import pandas as pd
from typing import Dict, List, Tuple, Any
from abc import ABC, abstractmethod
import json
import os
from common import Rule, RuleFactory, DataLoader

class Rule(ABC):
    """Abstract base class for all rule types."""

    def __init__(self, rule_config: Dict[str, Any]):
        """
        Initialize a rule from a configuration dictionary.

        Parameters:
        -----------
        rule_config : Dict[str, Any]
            The rule configuration from JSON
        """
        self.rule_type = rule_config.get('rule_type')
        self.rule = rule_config.get('rule')
        self.explanation = rule_config.get('explanation')
        self.columns = rule_config.get('columns', [])
        self.join_pairs = rule_config.get('join_pairs', [])
        self._validate_config()

    @abstractmethod
    def _validate_config(self):
        """Validate the rule configuration."""
        pass

    @abstractmethod
    def validate(self, dataset: Dict[str, pd.DataFrame]) -> Tuple[List[int], List[int]]:
        """
        Validate the rule against a dataset.

        Returns:
        --------
        Tuple[List[int], List[int]]
            - violating_rows: List of row indices where the rule is violated
            - valid_rows: List of row indices where the rule holds
        """
        pass

    def get_violation_report(self, dataset: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate a detailed violation report."""
        violating_rows, valid_rows = self.validate(dataset)
        total_rows = len(violating_rows) + len(valid_rows)
        violation_rate = len(violating_rows) / total_rows * 100 if total_rows > 0 else 0

        return {
            'rule_type': self.rule_type,
            'rule': self.rule,
            'total_rows': total_rows,
            'violating_rows': len(violating_rows),
            'valid_rows': len(valid_rows),
            'violation_rate': violation_rate,
            'violating_indices': violating_rows,
            'valid_indices': valid_rows
        }

    def save_validation_results(self, dataset: Dict[str, pd.DataFrame], output_dir: str = 'validation_results'):
        """
        Save validation results to CSV files.

        Parameters:
        -----------
        dataset : Dict[str, pd.DataFrame]
            Dictionary mapping table names to DataFrames
        output_dir : str
            Directory to save the CSV files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get validation results
        violating_rows, valid_rows = self.validate(dataset)

        # Create a safe filename from the rule
        rule_name = self.rule_type.replace(".", "_").replace(" ", "_").replace(",", "").replace("'", "")
        if len(rule_name) > 100:  # Limit filename length
            rule_name = rule_name[:100]

        # Determine the table and column being validated
        if self.join_pairs and len(self.join_pairs) > 0:
            join_pair = self.join_pairs[0]
            left_table = join_pair['left']['table']
            left_column = join_pair['left']['name']

            # Get the DataFrame
            df = dataset[left_table]

            # Save valid rows
            if valid_rows:
                valid_df = df.iloc[valid_rows].copy()
                valid_df['validation_status'] = 'valid'
                valid_path = os.path.join(output_dir, f"{rule_name}_valid.csv")
                valid_df.to_csv(valid_path, index=False)
                print(f"Saved {len(valid_rows)} valid rows to {valid_path}")

            # Save violating rows
            if violating_rows:
                violating_df = df.iloc[violating_rows].copy()
                violating_df['validation_status'] = 'violating'
                violating_path = os.path.join(output_dir, f"{rule_name}_violating.csv")
                violating_df.to_csv(violating_path, index=False)
                print(f"Saved {len(violating_rows)} violating rows to {violating_path}")


# class ReferentialIntegrityRule(Rule):
#     """Rule for checking referential integrity between tables."""
#
#     def _validate_config(self):
#         """Validate that the rule has the correct type and required fields."""
#         if self.rule_type != 'Referential Integrity':
#             raise ValueError(f"Expected rule_type 'Referential Integrity', got '{self.rule_type}'")
#
#         if not self.join_pairs:
#             raise ValueError("Referential Integrity rule must have at least one join_pair")
#
#         # Validate join_pair structure
#         for join_pair in self.join_pairs:
#             if 'left' not in join_pair or 'right' not in join_pair:
#                 raise ValueError("Each join_pair must have 'left' and 'right' fields")
#
#             for side in ['left', 'right']:
#                 if 'table' not in join_pair[side] or 'name' not in join_pair[side]:
#                     raise ValueError(f"join_pair['{side}'] must have 'table' and 'name' fields")
#
#     def validate(self, dataset: Dict[str, pd.DataFrame]) -> Tuple[List[int], List[int]]:
#         """
#         Validate referential integrity rule on the dataset.
#
#         Parameters:
#         -----------
#         dataset : Dict[str, pd.DataFrame]
#             Dictionary mapping table names to DataFrames
#
#         Returns:
#         --------
#         Tuple[List[int], List[int]]
#             - violating_rows: Indices where foreign key doesn't exist
#             - valid_rows: Indices where foreign key exists
#         """
#         # Get the first join pair (for simple referential integrity)
#         join_pair = self.join_pairs[0]
#         left = join_pair['left']
#         right = join_pair['right']
#
#         # Extract table and column names
#         left_table = left['table']
#         left_column = left['name']
#         right_table = right['table']
#         right_column = right['name']
#
#         # Validate tables exist
#         if left_table not in dataset:
#             raise KeyError(f"Table '{left_table}' not found in dataset")
#         if right_table not in dataset:
#             raise KeyError(f"Table '{right_table}' not found in dataset")
#
#         # Get DataFrames
#         left_df = dataset[left_table]
#         right_df = dataset[right_table]
#
#         # Validate columns exist
#         if left_column not in left_df.columns:
#             raise KeyError(f"Column '{left_column}' not found in table '{left_table}'")
#         if right_column not in right_df.columns:
#             raise KeyError(f"Column '{right_column}' not found in table '{right_table}'")
#
#         # Get valid reference values
#         valid_references = set(right_df[right_column].dropna().unique())
#
#         # Check each row
#         violating_rows = []
#         valid_rows = []
#
#         for idx, value in enumerate(left_df[left_column]):
#             if pd.isna(value) or value is None:
#                 violating_rows.append(idx)
#             elif value in valid_references:
#                 valid_rows.append(idx)
#             else:
#                 violating_rows.append(idx)
#
#         return violating_rows, valid_rows


class CrossTableConsistencyRule(Rule):
    """Rule for checking consistency of values across tables when joined on a key."""

    def _validate_config(self):
        """Validate that the rule has the correct type and required fields."""
        if self.rule_type != 'Cross-Table Consistency':
            raise ValueError(f"Expected rule_type 'Cross-Table Consistency', got '{self.rule_type}'")

        if not self.join_pairs:
            raise ValueError("Cross-Table Consistency rule must have at least one join_pair")

        if not self.columns or len(self.columns) < 2:
            raise ValueError("Cross-Table Consistency rule must have at least 2 columns to compare")

        # Validate join_pair and columns structure
        for join_pair in self.join_pairs:
            if 'left' not in join_pair or 'right' not in join_pair:
                raise ValueError("Each join_pair must have 'left' and 'right' fields")

    def validate(self, dataset: Dict[str, pd.DataFrame]) -> Tuple[List[int], List[int]]:
        """
        Validate cross-table consistency rule on the dataset using vectorized operations.

        This rule checks: IF table1.join_col = table2.join_col THEN table1.check_col = table2.check_col

        Parameters:
        -----------
        dataset : Dict[str, pd.DataFrame]
            Dictionary mapping table names to DataFrames

        Returns:
        --------
        Tuple[List[int], List[int]]
            - violating_rows: Indices in left table where values don't match after join
            - valid_rows: Indices in left table where values match or no join exists
        """
        # Get join information
        join_pair = self.join_pairs[0]
        left_join = join_pair['left']
        right_join = join_pair['right']

        # Get tables and join columns
        left_table = left_join['table']
        left_join_col = left_join['name']
        right_table = right_join['table']
        right_join_col = right_join['name']

        # Get comparison columns (the columns that should be consistent)
        left_check_col = None
        right_check_col = None

        for col in self.columns:
            if col['table'] == left_table:
                left_check_col = col['name']
            elif col['table'] == right_table:
                right_check_col = col['name']

        if not left_check_col or not right_check_col:
            raise ValueError("Could not identify columns to check for consistency")

        # Validate tables exist
        if left_table not in dataset:
            raise KeyError(f"Table '{left_table}' not found in dataset")
        if right_table not in dataset:
            raise KeyError(f"Table '{right_table}' not found in dataset")

        # Get DataFrames
        left_df = dataset[left_table].copy()
        right_df = dataset[right_table].copy()

        # Add index column to track original rows
        left_df['_original_index'] = range(len(left_df))

        # Perform inner join to find matching records
        merged_df = pd.merge(
            left_df,
            right_df,
            left_on=left_join_col,
            right_on=right_join_col,
            suffixes=('_left', '_right'),
            how='inner'
        )

        # Get the check columns with their suffixes
        left_check_col_merged = f'{left_check_col}_left' if f'{left_check_col}_left' in merged_df.columns else left_check_col
        right_check_col_merged = f'{right_check_col}_right' if f'{right_check_col}_right' in merged_df.columns else right_check_col

        # Create a mask for rows where values are consistent
        # Consider NaN == NaN as True
        merged_df['is_consistent'] = (
            (merged_df[left_check_col_merged].isna() & merged_df[right_check_col_merged].isna()) |
            (merged_df[left_check_col_merged] == merged_df[right_check_col_merged])
        )

        # Group by original index and check if all matches are consistent
        consistency_by_index = merged_df.groupby('_original_index')['is_consistent'].all()

        # Get all indices from left table
        all_indices = set(range(len(left_df)))
        
        # Get indices that had matches
        matched_indices = set(consistency_by_index.index)
        
        # Valid rows are those that either:
        # 1. Had no matches (no consistency to check)
        # 2. Had matches and all were consistent
        valid_indices = list(
            (all_indices - matched_indices) |  # No matches
            set(consistency_by_index[consistency_by_index].index)  # All matches consistent
        )
        
        # Violating rows are those that had matches but some were inconsistent
        violating_indices = list(
            set(consistency_by_index[~consistency_by_index].index)
        )

        return violating_indices, valid_indices


# Register the Cross-Table Consistency rule type
RuleFactory.register_rule_type('Cross-Table Consistency', CrossTableConsistencyRule)
