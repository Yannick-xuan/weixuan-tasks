import pandas as pd
from typing import Dict, List, Tuple, Any
import json
import os
from common import Rule, RuleFactory, DataLoader


class ReferentialIntegrityRule(Rule):
    """Rule for checking referential integrity between tables."""

    def _validate_config(self):
        """Validate that the rule has the correct type and required fields."""
        if self.rule_type != 'Referential Integrity':
            raise ValueError(f"Expected rule_type 'Referential Integrity', got '{self.rule_type}'")

        if not self.join_pairs:
            raise ValueError("Referential Integrity rule must have at least one join_pair")

        # Validate join_pair structure
        for join_pair in self.join_pairs:
            if 'left' not in join_pair or 'right' not in join_pair:
                raise ValueError("Each join_pair must have 'left' and 'right' fields")

            for side in ['left', 'right']:
                if 'table' not in join_pair[side] or 'name' not in join_pair[side]:
                    raise ValueError(f"join_pair['{side}'] must have 'table' and 'name' fields")

    def validate(self, dataset: Dict[str, pd.DataFrame]) -> Tuple[List[int], List[int]]:
        """
        Validate referential integrity rule on the dataset using vectorized operations.

        Parameters:
        -----------
        dataset : Dict[str, pd.DataFrame]
            Dictionary mapping table names to DataFrames

        Returns:
        --------
        Tuple[List[int], List[int]]
            - violating_rows: Indices where foreign key doesn't exist
            - valid_rows: Indices where foreign key exists
        """
        # Get the first join pair (for simple referential integrity)
        join_pair = self.join_pairs[0]
        left = join_pair['left']
        right = join_pair['right']

        # Extract table and column names
        left_table = left['table']
        left_column = left['name']
        right_table = right['table']
        right_column = right['name']

        # Validate tables exist
        if left_table not in dataset:
            raise KeyError(f"Table '{left_table}' not found in dataset")
        if right_table not in dataset:
            raise KeyError(f"Table '{right_table}' not found in dataset")

        # Get DataFrames
        left_df = dataset[left_table]
        right_df = dataset[right_table]

        # Validate columns exist
        if left_column not in left_df.columns:
            raise KeyError(f"Column '{left_column}' not found in table '{left_table}'")
        if right_column not in right_df.columns:
            raise KeyError(f"Column '{right_column}' not found in table '{right_table}'")

        # Get valid reference values as a set for fast lookups
        valid_references = set(right_df[right_column].dropna().unique())

        # Create masks for valid and invalid rows
        # First, handle null values
        null_mask = left_df[left_column].isna()
        
        # Then check if non-null values exist in valid_references
        non_null_mask = ~null_mask
        valid_ref_mask = left_df[left_column].isin(valid_references)
        
        # Combine masks to get final valid and violating rows
        valid_mask = null_mask | (non_null_mask & valid_ref_mask)
        violating_mask = ~valid_mask

        # Convert masks to indices
        valid_rows = left_df.index[valid_mask].tolist()
        violating_rows = left_df.index[violating_mask].tolist()

        return violating_rows, valid_rows


# Register the rule type with the factory
RuleFactory.register_rule_type('Referential Integrity', ReferentialIntegrityRule)