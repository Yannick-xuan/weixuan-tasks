import pandas as pd
from typing import Dict, List, Tuple, Any
from abc import ABC, abstractmethod
import json
import os

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
        Validate cross-table consistency rule on the dataset.

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

        violating_rows = []
        valid_rows = []

        # Check all rows in the left table
        for idx in range(len(left_df)):
            # Find this row in the merged dataframe
            matched_rows = merged_df[merged_df['_original_index'] == idx]

            if len(matched_rows) == 0:
                # No join match found - this is considered valid (no consistency to check)
                valid_rows.append(idx)
            else:
                # Check if values are consistent across all matches
                is_consistent = True

                for _, row in matched_rows.iterrows():
                    # Get values from both sides
                    left_value = row.get(f'{left_check_col}_left', row.get(left_check_col))
                    right_value = row.get(f'{right_check_col}_right', row.get(right_check_col))

                    # Handle NaN values - consider NaN == NaN as True
                    if pd.isna(left_value) and pd.isna(right_value):
                        continue
                    elif pd.isna(left_value) or pd.isna(right_value):
                        is_consistent = False
                        break
                    elif left_value != right_value:
                        is_consistent = False
                        break

                if is_consistent:
                    valid_rows.append(idx)
                else:
                    violating_rows.append(idx)

        return violating_rows, valid_rows


class RuleFactory:
    """Factory class for creating rule instances from configuration."""

    _rule_classes = {
        # 'Referential Integrity': ReferentialIntegrityRule,
        'Cross-Table Consistency': CrossTableConsistencyRule,
        # Add more rule types here as they are implemented
    }

    @classmethod
    def create_rule(cls, rule_config: Dict[str, Any]) -> Rule:
        """
        Create a rule instance from configuration.

        Parameters:
        -----------
        rule_config : Dict[str, Any]
            Rule configuration from JSON

        Returns:
        --------
        Rule
            An instance of the appropriate Rule subclass
        """
        rule_type = rule_config.get('rule_type')

        if rule_type not in cls._rule_classes:
            raise ValueError(f"Unknown rule type: '{rule_type}'")

        rule_class = cls._rule_classes[rule_type]
        return rule_class(rule_config)


class DataLoader:
    """Class for loading datasets from Excel files."""

    @staticmethod
    def load_datasets(data_dir: str = '.') -> Dict[str, pd.DataFrame]:
        """
        Load all datasets from Excel files.

        Parameters:
        -----------
        data_dir : str
            Directory containing the Excel files

        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping table names to DataFrames
        """
        dataset = {}

        # Define file mappings
        file_mappings = {
            'uo': 'unita-organizzative.xlsx',
            'enti': 'enti.xlsx',
            'uo_sfe': 'unita-organizzative-che-ricevono-fatture-elettroniche.xlsx',
            'responsabili': 'responsabili-della-transizione-al-digitale.xlsx'
        }

        # Load each file
        for table_name, filename in file_mappings.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                print(f"Loading {table_name} from {filename}...")
                dataset[table_name] = pd.read_excel(filepath)
                print(f"  Loaded {len(dataset[table_name])} rows")
            else:
                print(f"Warning: File {filepath} not found")

        return dataset




# Import the classes from your provided code
# Assuming the classes are saved in a file called 'rule_validator.py'
# from rule_validator import Rule, CrossTableConsistencyRule, RuleFactory, DataLoader

def main():
    """Main function to validate Cross-Table Consistency rules on real datasets."""

    # Load the datasets
    print("Loading datasets...")
    dataset = DataLoader.load_datasets()

    if not dataset:
        print("Error: No datasets loaded. Please ensure Excel files are in the current directory.")
        return

    # Print dataset info
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    for table_name, df in dataset.items():
        print(f"{table_name}: {len(df)} rows, {len(df.columns)} columns")

    # Define Cross-Table Consistency rules
    cross_table_rules = [
        {
            "rule_type": "Cross-Table Consistency",
            "rule": "IF uo.Codice_uni_uo = responsabili.Codice_uni_uo THEN uo.Descrizione_uo = responsabili.Descrizione_uo",
            "explanation": "When the same organizational unit is referenced in both tables, the unit description must be consistent.",
            "columns": [
                {"table": "uo", "name": "Descrizione_uo"},
                {"table": "responsabili", "name": "Descrizione_uo"}
            ],
            "join_pairs": [
                {
                    "left": {"table": "uo", "name": "Codice_uni_uo"},
                    "right": {"table": "responsabili", "name": "Codice_uni_uo"}
                }
            ]
        }
    ]

    # Process each rule
    print("\n" + "=" * 70)
    print("CROSS-TABLE CONSISTENCY VALIDATION RESULTS")
    print("=" * 70)

    all_results = []

    for i, rule_config in enumerate(cross_table_rules):
        print(f"\nRule {i + 1}: {rule_config['rule']}")
        print("-" * 70)

        try:
            # Create rule instance
            rule = RuleFactory.create_rule(rule_config)

            # Validate
            violating_rows, valid_rows = rule.validate(dataset)

            # Get report
            report = rule.get_violation_report(dataset)
            all_results.append(report)

            # Print results
            print(f"Total rows checked: {report['total_rows']}")
            print(f"Valid rows: {report['valid_rows']}")
            print(f"Violating rows: {report['violating_rows']}")
            print(f"Violation rate: {report['violation_rate']:.2f}%")

            # Show sample violations if any
            if violating_rows and len(violating_rows) > 0:
                print(f"\nSample violations (showing first 3):")

                # Get the tables and columns involved
                join_pair = rule_config['join_pairs'][0]
                left_table = join_pair['left']['table']
                left_join_col = join_pair['left']['name']
                right_table = join_pair['right']['table']
                right_join_col = join_pair['right']['name']

                # Get check columns
                left_check_col = next(col['name'] for col in rule_config['columns'] if col['table'] == left_table)
                right_check_col = next(col['name'] for col in rule_config['columns'] if col['table'] == right_table)

                # Show violations
                left_df = dataset[left_table]
                right_df = dataset[right_table]

                for idx in violating_rows[:3]:
                    left_row = left_df.iloc[idx]
                    join_value = left_row[left_join_col]

                    # Find matching row in right table
                    matching_rows = right_df[right_df[right_join_col] == join_value]

                    if not matching_rows.empty:
                        right_row = matching_rows.iloc[0]
                        print(f"\n  Row {idx}:")
                        print(f"    Join key ({left_join_col}): {join_value}")
                        print(f"    {left_table}.{left_check_col}: '{left_row[left_check_col]}'")
                        print(f"    {right_table}.{right_check_col}: '{right_row[right_check_col]}'")

        except Exception as e:
            print(f"Error validating rule: {e}")
            import traceback
            traceback.print_exc()

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    if all_results:
        total_rules = len(all_results)
        total_rows_checked = sum(r['total_rows'] for r in all_results)
        total_violations = sum(r['violating_rows'] for r in all_results)
        avg_violation_rate = sum(r['violation_rate'] for r in all_results) / total_rules

        print(f"Total rules validated: {total_rules}")
        print(f"Total rows checked across all rules: {total_rows_checked}")
        print(f"Total violations found: {total_violations}")
        print(f"Average violation rate: {avg_violation_rate:.2f}%")

        # Find rules with highest violation rates
        print("\nRules with highest violation rates:")
        sorted_results = sorted(all_results, key=lambda x: x['violation_rate'], reverse=True)
        for i, result in enumerate(sorted_results[:3]):
            print(f"  {i + 1}. {result['rule'][:60]}...")
            print(f"     Violation rate: {result['violation_rate']:.2f}%")

    # Export results to CSV
    print("\n" + "=" * 70)
    print("EXPORTING RESULTS")
    print("=" * 70)

    # 输出目录
    output_dir = 'validation_results'
    os.makedirs(output_dir, exist_ok=True)

    # # Create a summary DataFrame
    # summary_data = []
    # for i, (rule_config, result) in enumerate(zip(cross_table_rules, all_results)):
    #     summary_data.append({
    #         'Rule_Number': i + 1,
    #         'Rule_Type': result['rule_type'],
    #         'Rule_Description': rule_config['rule'],
    #         'Total_Rows': result['total_rows'],
    #         'Valid_Rows': result['valid_rows'],
    #         'Violating_Rows': result['violating_rows'],
    #         'Violation_Rate_%': round(result['violation_rate'], 2)
    #     })
    #
    # summary_df = pd.DataFrame(summary_data)
    # summary_file = os.path.join(output_dir, 'CrossTableConsis_summary.csv')
    # summary_df.to_csv(summary_file, index=False)
    # print(f"Results exported to: {summary_file}")

    # Also save detailed violation and valid indices for each rule
    for i, (rule_config, result) in enumerate(zip(cross_table_rules, all_results)):
        join_pair = rule_config['join_pairs'][0]
        left_table = join_pair['left']['table']
        left_df = dataset[left_table]
        # violation
        if result['violating_indices']:
            print(result['violating_indices'])
            violation_rows_df = left_df.iloc[result['violating_indices']].copy()
            violation_rows_df['validation_status'] = 'violation'
            violations_file = os.path.join(output_dir, f'CrossTableConsis_violation_rule_{i + 1}.csv')
            violation_rows_df.to_csv(violations_file, index=False)
            print(f"Violation rows for rule {i + 1} saved to: {violations_file}")
        # valid
        if result['valid_indices']:
            valid_rows_df = left_df.iloc[result['valid_indices']].copy()
            print(result['valid_indices'])
            valid_rows_df['validation_status'] = 'valid'
            valid_file = os.path.join(output_dir, f'CrossTableConsis_valid_rule_{i + 1}.csv')
            valid_rows_df.to_csv(valid_file, index=False)
            print(f"Valid rows for rule {i + 1} saved to: {valid_file}")


if __name__ == "__main__":
    # Add the provided classes here or import them from rule_validator.py
    main()
