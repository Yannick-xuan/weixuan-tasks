"""
Implementation for Conditional Reference Existence rules
"""
import pandas as pd
from typing import Dict, List, Tuple, Any
import re
import os
from common import Rule, RuleFactory, DataLoader


class ConditionalReferenceExistence(Rule):
    """Rule for checking conditional reference integrity based on label/text conditions."""

    def _validate_config(self):
        """Validate that the rule has the correct type and required fields."""
        if self.rule_type != 'Conditional Reference Existence':
            raise ValueError(f"Expected rule_type 'Conditional Reference Existence', got '{self.rule_type}'")

        if not self.join_pairs:
            raise ValueError("Conditional Reference rule must have at least one join_pair")

        if not self.columns or len(self.columns) < 2:
            raise ValueError("Conditional Reference rule must have at least 2 columns")

        if 'contains' not in self.rule and 'CONTAINS' not in self.rule and '=' not in self.rule:
            raise ValueError("Conditional Reference rule must contain either 'contains' or '=' operator in the rule text")

    def _parse_condition(self) -> Dict[str, Any]:
        """Parse the rule text to extract the condition and reference requirement."""
        # Pattern 1: IF table.column contains 'text' THEN table.column must exist in table.column
        pattern1 = r"IF\s+(\w+)\.(\w+)\s+contains\s+'([^']+)'\s+THEN\s+(\w+)\.(\w+)\s+must\s+exist\s+in\s+(\w+)\.(\w+)"
        # Pattern 2: IF table.column = 'text' THEN table.column must exist in table.column
        pattern2 = r"IF\s+(\w+)\.(\w+)\s+=\s+'([^']+)'\s+THEN\s+(\w+)\.(\w+)\s+must\s+exist\s+in\s+(\w+)\.(\w+)"
        
        match1 = re.search(pattern1, self.rule, re.IGNORECASE)
        match2 = re.search(pattern2, self.rule, re.IGNORECASE)
        
        match = match1 or match2
        if not match:
            raise ValueError(f"Could not parse rule: {self.rule}")

        return {
            'condition_table': match.group(1),
            'condition_column': match.group(2),
            'search_text': match.group(3),
            'check_table': match.group(4),
            'check_column': match.group(5),
            'reference_table': match.group(6),
            'reference_column': match.group(7),
            'operator': 'contains' if match1 else '='
        }

    def validate(self, dataset: Dict[str, pd.DataFrame]) -> Tuple[List[int], List[int]]:
        """
        Validate conditional reference rule on the dataset using vectorized operations.

        This rule checks: IF table1.col1 contains/equals 'text' THEN table1.col2 must exist in table2.col3
        """
        # Parse the condition
        condition = self._parse_condition()

        # Get tables
        condition_table = condition['condition_table']
        reference_table = condition['reference_table']

        # Validate tables exist
        if condition_table not in dataset:
            raise KeyError(f"Table '{condition_table}' not found in dataset")
        if reference_table not in dataset:
            raise KeyError(f"Table '{reference_table}' not found in dataset")

        # Get DataFrames
        cond_df = dataset[condition_table]
        ref_df = dataset[reference_table]

        # Get columns
        condition_col = condition['condition_column']
        check_col = condition['check_column']
        reference_col = condition['reference_column']
        search_text = condition['search_text'].lower()
        operator = condition['operator']

        # Validate columns exist
        if condition_col not in cond_df.columns:
            raise KeyError(f"Column '{condition_col}' not found in table '{condition_table}'")
        if check_col not in cond_df.columns:
            raise KeyError(f"Column '{check_col}' not found in table '{condition_table}'")
        if reference_col not in ref_df.columns:
            raise KeyError(f"Column '{reference_col}' not found in table '{reference_table}'")

        # Get valid reference values as a set for fast lookups
        valid_references = set(ref_df[reference_col].dropna().unique())

        # Create masks for validation
        # 1. Check if condition is met (contains/equals the search text)
        condition_values = cond_df[condition_col].astype(str).str.lower()
        if operator == 'contains':
            condition_met_mask = condition_values.str.contains(search_text, na=False)
        else:  # operator == '='
            condition_met_mask = condition_values == search_text
        
        # 2. Check if reference exists for rows where condition is met
        check_values = cond_df[check_col]
        valid_ref_mask = check_values.isin(valid_references)
        
        # 3. Combine masks to get final valid and violating rows
        # Rows are valid if:
        # - condition is not met (no need to check reference)
        # - condition is met AND reference exists
        valid_mask = ~condition_met_mask | (condition_met_mask & valid_ref_mask)
        violating_mask = ~valid_mask

        # Convert masks to indices
        valid_rows = cond_df.index[valid_mask].tolist()
        violating_rows = cond_df.index[violating_mask].tolist()

        return violating_rows, valid_rows


# Register the rule type with the factory
RuleFactory.register_rule_type('Conditional Reference Existence', ConditionalReferenceExistence)