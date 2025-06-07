"""
Complete implementation for Enhanced Conditional Reference with Label Restriction rules
Supports both forward and reverse conditional references without hardcoding
"""
import pandas as pd
from typing import Dict, List, Tuple, Any
import re
from abc import ABC, abstractmethod
import os
import json


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
        elif match2:
            return {
                'type': 'reverse',
                'left_table': match2.group(1),
                'left_column': match2.group(2),
                'right_table': match2.group(3),
                'right_column': match2.group(4),
                'check_table': match2.group(5),
                'check_column': match2.group(6),
                'search_text': match2.group(7)
            }
        else:
            raise ValueError(f"Could not parse rule: {self.rule}")

    def validate(self, dataset: Dict[str, pd.DataFrame]) -> Tuple[List[int], List[int]]:
        """Validate conditional reference rule on the dataset."""
        condition = self._parse_condition()

        if condition['type'] == 'forward':
            return self._validate_forward(dataset, condition)
        else:
            return self._validate_reverse(dataset, condition)

    def _validate_forward(self, dataset: Dict[str, pd.DataFrame], condition: Dict[str, Any]) -> Tuple[
        List[int], List[int]]:
        """Original forward validation logic."""
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

        # Get valid reference values
        valid_references = set(ref_df[reference_col].dropna().unique())

        violating_rows = []
        valid_rows = []

        # Check each row
        for idx in range(len(cond_df)):
            condition_value = cond_df.iloc[idx][condition_col]
            check_value = cond_df.iloc[idx][check_col]

            if pd.notna(condition_value) and search_text in str(condition_value).lower():
                if pd.isna(check_value) or check_value not in valid_references:
                    violating_rows.append(idx)
                else:
                    valid_rows.append(idx)
            else:
                valid_rows.append(idx)

        return violating_rows, valid_rows

    def _validate_reverse(self, dataset: Dict[str, pd.DataFrame], condition: Dict[str, Any]) -> Tuple[
        List[int], List[int]]:
        """
        Reverse validation: IF table1.col1 = table2.col2 THEN table2.col3 contains 'text'
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

        # Validate columns exist
        if left_col not in left_df.columns:
            raise KeyError(f"Column '{left_col}' not found in table '{left_table}'")
        if right_col not in right_df.columns:
            raise KeyError(f"Column '{right_col}' not found in table '{right_table}'")
        if check_col not in right_df.columns:
            raise KeyError(f"Column '{check_col}' not found in table '{right_table}'")

        violating_rows = []
        valid_rows = []

        # For each row in left table (responsabili)
        for idx in range(len(left_df)):
            left_value = left_df.iloc[idx][left_col]

            if pd.isna(left_value):
                # No reference - consider valid
                valid_rows.append(idx)
                continue

            # Find matching rows in right table
            matching_rows = right_df[right_df[right_col] == left_value]

            if len(matching_rows) == 0:
                # No match found - this might be a referential integrity issue
                # For this rule, we consider it a violation
                violating_rows.append(idx)
            else:
                # Check if any matching row contains the search text
                contains_text = False
                for _, row in matching_rows.iterrows():
                    check_value = row[check_col]
                    if pd.notna(check_value) and search_text in str(check_value).lower():
                        contains_text = True
                        break

                if contains_text:
                    valid_rows.append(idx)
                else:
                    violating_rows.append(idx)

        return violating_rows, valid_rows


class RuleFactory:
    """Factory class for creating rule instances from configuration."""

    _rule_classes = {
        'Conditional Reference with Label Restriction': ConditionalReferenceWithLabelRestrictionRule,
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


def main():
    """Main function to validate Conditional Reference with Label Restriction rules on real datasets."""

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

    # Define Conditional Reference with Label Restriction rules
    conditional_rules = [
        {
            "rule_type": "Conditional Reference with Label Restriction",
            "rule": "IF responsabili.Codice_uni_uo = uo.Codice_uni_uo THEN uo.Descrizione_uo contains 'transizione al digitale'",
            "explanation": "When a Digital Transition Manager references an organizational unit, that unit in the Organizational Units table must contain 'digital transition' in its description. This ensures managers are always associated with dedicated digital transition offices, not unrelated units.",
            "columns": [
                {"table": "responsabili", "name": "Codice_uni_uo"},
                {"table": "uo", "name": "Descrizione_uo"}
            ],
            "join_pairs": [
                {
                    "left": {"table": "responsabili", "name": "Codice_uni_uo"},
                    "right": {"table": "uo", "name": "Codice_uni_uo"}
                }
            ]
        }
    ]

    # Process each rule
    print("\n" + "=" * 70)
    print("CONDITIONAL REFERENCE WITH LABEL RESTRICTION VALIDATION RESULTS")
    print("=" * 70)

    all_results = []

    for i, rule_config in enumerate(conditional_rules):
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

                # Parse the condition to understand the rule type
                condition = rule._parse_condition()

                if condition['type'] == 'forward':
                    # Forward rule: show rows that meet condition but fail reference
                    table_name = condition['condition_table']
                    df = dataset[table_name]

                    for idx in violating_rows[:3]:
                        row = df.iloc[idx]
                        print(f"\n  Row {idx}:")
                        print(f"    {condition['condition_column']}: {row[condition['condition_column']]}")
                        print(f"    {condition['check_column']}: {row[condition['check_column']]}")

                else:
                    # Reverse rule: show responsabili that don't reference proper units
                    left_table = condition['left_table']
                    right_table = condition['right_table']
                    left_df = dataset[left_table]
                    right_df = dataset[right_table]

                    for idx in violating_rows[:3]:
                        left_row = left_df.iloc[idx]
                        left_value = left_row[condition['left_column']]

                        print(f"\n  Row {idx}:")
                        if 'Nome_responsabile' in left_row and 'Cognome_responsabile' in left_row:
                            print(
                                f"    Responsabile: {left_row['Nome_responsabile']} {left_row['Cognome_responsabile']}")
                        print(f"    {condition['left_column']}: {left_value}")

                        # Find matching UO
                        matching_rows = right_df[right_df[condition['right_column']] == left_value]
                        if not matching_rows.empty:
                            desc = matching_rows.iloc[0][condition['check_column']]
                            print(f"    UO Description: {desc[:80]}...")
                            print(f"    Contains '{condition['search_text']}': NO")
                        else:
                            print(f"    No matching unit found in {right_table}")

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

    output_dir = 'validation_results'
    os.makedirs(output_dir, exist_ok=True)

    # # Create a summary DataFrame
    # summary_data = []
    # for i, (rule_config, result) in enumerate(zip(conditional_rules, all_results)):
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
    # summary_file = os.path.join(output_dir, 'ConditionalReferenceWithLabelRestriction_summary.csv')
    # summary_df.to_csv(summary_file, index=False)
    # print(f"Results exported to: {summary_file}")

    # valid/violation
    for i, (rule_config, result) in enumerate(zip(conditional_rules, all_results)):
        join_pair = rule_config['join_pairs'][0]
        left_table = join_pair['left']['table']
        if left_table in dataset:
            left_df = dataset[left_table]
            # valid
            if result['valid_indices']:
                print(result['valid_indices'])
                valid_rows_df = left_df.iloc[result['valid_indices']].copy()
                valid_rows_df['validation_status'] = 'valid'
                valid_file = os.path.join(output_dir, f'ConditionalReferenceWithLabelRestriction_valid_rule_{i+1}.csv')
                valid_rows_df.to_csv(valid_file, index=False)
                print(f"Valid rows for rule {i+1} saved to: {valid_file}")
            # violation
            if result['violating_indices']:
                print(result['violating_indices'])
                violation_rows_df = left_df.iloc[result['violating_indices']].copy()
                violation_rows_df['validation_status'] = 'violation'
                violation_file = os.path.join(output_dir, f'ConditionalReferenceWithLabelRestriction_violation_rule_{i+1}.csv')
                violation_rows_df.to_csv(violation_file, index=False)
                print(f"Violation rows for rule {i+1} saved to: {violation_file}")

if __name__ == "__main__":
    main()