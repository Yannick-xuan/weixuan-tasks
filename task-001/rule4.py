"""
Implementation for Enhanced Conditional Reference with Label Restriction rules
Supports both forward and reverse conditional references without hardcoding
"""
import pandas as pd
from typing import Dict, List, Tuple, Any
import re
import os
from common import Rule, RuleFactory, DataLoader


class ConditionalReferenceWithLabelRestrictionRule(Rule):
    """Enhanced rule for checking both forward and reverse conditional reference integrity."""

    def _validate_config(self):
        """Validate that the rule has the correct type and required fields."""
        if self.rule_type != 'Conditional Reference with Label Restriction':
            raise ValueError(
                f"Expected rule_type 'Conditional Reference with Label Restriction', got '{self.rule_type}'")

        if not self.join_pairs:
            raise ValueError("Conditional Reference rule must have at least one join_pair")

        if not self.columns or len(self.columns) < 2:
            raise ValueError("Conditional Reference rule must have at least 2 columns")

    def _parse_condition(self) -> Dict[str, Any]:
        """Parse the rule text to extract the condition and reference requirement."""
        # Pattern 1: IF table.column contains 'text' THEN table.column must exist in table.column
        pattern1 = r"IF\s+(\w+)\.(\w+)\s+contains\s+'([^']+)'\s+THEN\s+(\w+)\.(\w+)\s+must\s+exist\s+in\s+(\w+)\.(\w+)"
        match1 = re.search(pattern1, self.rule, re.IGNORECASE)

        # Pattern 2: IF table1.col1 = table2.col2 THEN table2.col3 contains 'text'
        pattern2 = r"IF\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)\s+THEN\s+(\w+)\.(\w+)\s+contains\s+'([^']+)'"
        match2 = re.search(pattern2, self.rule, re.IGNORECASE)

        # Pattern 3: IF table1.col1 = table2.col2 THEN table2.col3 = 'text'
        pattern3 = r"IF\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)\s+THEN\s+(\w+)\.(\w+)\s+=\s+'([^']+)'"
        match3 = re.search(pattern3, self.rule, re.IGNORECASE)

        if match1:
            return {
                'type': 'forward',
                'condition_table': match1.group(1),
                'condition_column': match1.group(2),
                'search_text': match1.group(3),
                'check_table': match1.group(4),
                'check_column': match1.group(5),
                'reference_table': match1.group(6),
                'reference_column': match1.group(7)
            }
        elif match2 or match3:
            match = match2 or match3
            return {
                'type': 'reverse',
                'left_table': match.group(1),
                'left_column': match.group(2),
                'right_table': match.group(3),
                'right_column': match.group(4),
                'check_table': match.group(5),
                'check_column': match.group(6),
                'search_text': match.group(7),
                'operator': 'contains' if match2 else '='
            }
        else:
            raise ValueError(f"Could not parse rule: {self.rule}")

    def validate(self, dataset: Dict[str, pd.DataFrame]) -> Tuple[List[int], List[int]]:
        """Validate conditional reference rule on the dataset using vectorized operations."""
        condition = self._parse_condition()

        if condition['type'] == 'forward':
            return self._validate_forward(dataset, condition)
        else:
            return self._validate_reverse(dataset, condition)

    def _validate_forward(self, dataset: Dict[str, pd.DataFrame], condition: Dict[str, Any]) -> Tuple[
        List[int], List[int]]:
        """Original forward validation logic using vectorized operations."""
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
        # 1. Check if condition is met (contains the search text)
        condition_values = cond_df[condition_col].astype(str).str.lower()
        condition_met_mask = condition_values.str.contains(search_text, na=False)
        
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

    def _validate_reverse(self, dataset: Dict[str, pd.DataFrame], condition: Dict[str, Any]) -> Tuple[
        List[int], List[int]]:
        """
        Reverse validation using vectorized operations: IF table1.col1 = table2.col2 THEN table2.col3 contains/equals 'text'
        This checks rows in table1 (responsabili) and validates against table2 (uo)
        """
        # Get tables
        left_table = condition['left_table']
        right_table = condition['right_table']

        # Validate tables exist
        if left_table not in dataset:
            raise KeyError(f"Table '{left_table}' not found in dataset")
        if right_table not in dataset:
            raise KeyError(f"Table '{right_table}' not found in dataset")

        # Get DataFrames
        left_df = dataset[left_table]
        right_df = dataset[right_table]

        # Get columns
        left_col = condition['left_column']
        right_col = condition['right_column']
        check_col = condition['check_column']
        search_text = condition['search_text'].lower()
        operator = condition.get('operator', 'contains')  # Default to 'contains' for backward compatibility

        # Validate columns exist
        if left_col not in left_df.columns:
            raise KeyError(f"Column '{left_col}' not found in table '{left_table}'")
        if right_col not in right_df.columns:
            raise KeyError(f"Column '{right_col}' not found in table '{right_table}'")
        if check_col not in right_df.columns:
            raise KeyError(f"Column '{check_col}' not found in table '{right_table}'")

        # Create a mapping of right_col values to their check_col values
        right_df['_check_value'] = right_df[check_col].astype(str).str.lower()
        if operator == 'contains':
            right_df['_contains_text'] = right_df['_check_value'].str.contains(search_text, na=False)
        else:  # operator == '='
            right_df['_contains_text'] = right_df['_check_value'] == search_text
        value_to_contains = dict(zip(right_df[right_col], right_df['_contains_text']))

        # Create masks for validation
        # 1. Rows with no reference are considered valid
        no_ref_mask = left_df[left_col].isna()
        
        # 2. For rows with references, check if any matching row contains/equals the text
        has_ref_mask = ~no_ref_mask
        left_values = left_df[left_col]
        
        # Create a mask for rows where the reference exists and contains/equals the text
        ref_exists_mask = left_values.isin(value_to_contains.keys())
        ref_contains_text_mask = left_values.map(lambda x: value_to_contains.get(x, False))
        
        # Combine masks to get final valid and violating rows
        valid_mask = no_ref_mask | (has_ref_mask & ref_exists_mask & ref_contains_text_mask)
        violating_mask = ~valid_mask

        # Convert masks to indices
        valid_rows = left_df.index[valid_mask].tolist()
        violating_rows = left_df.index[violating_mask].tolist()

        return violating_rows, valid_rows


# Register the rule type with the factory
RuleFactory.register_rule_type('Conditional Reference with Label Restriction', ConditionalReferenceWithLabelRestrictionRule)