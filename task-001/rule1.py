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
                print(valid_rows)
                valid_df['validation_status'] = 'valid'
                valid_path = os.path.join(output_dir, f"{rule_name}_valid.csv")
                valid_df.to_csv(valid_path, index=False)
                print(f"Saved {len(valid_rows)} valid rows to {valid_path}")

            # Save violating rows
            if violating_rows:
                print(violating_rows)
                violating_df = df.iloc[violating_rows].copy()
                violating_df['validation_status'] = 'violating'
                violating_path = os.path.join(output_dir, f"{rule_name}_violating.csv")
                violating_df.to_csv(violating_path, index=False)
                print(f"Saved {len(violating_rows)} violating rows to {violating_path}")

            # # Save summary
            # summary = {
            #     'rule_type': [self.rule_type],
            #     'rule': [self.rule],
            #     'total_rows': [len(valid_rows) + len(violating_rows)],
            #     'valid_rows': [len(valid_rows)],
            #     'violating_rows': [len(violating_rows)],
            #     'violation_rate': [len(violating_rows) / (len(valid_rows) + len(violating_rows)) * 100 if (
            #                                                                                                           len(valid_rows) + len(
            #                                                                                                       violating_rows)) > 0 else 0]
            # }
            # summary_df = pd.DataFrame(summary)
            # summary_path = os.path.join(output_dir, f"{rule_name}_summary.csv")
            # summary_df.to_csv(summary_path, index=False)
            # print(f"Saved validation summary to {summary_path}")
            #

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
        Validate referential integrity rule on the dataset.

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

        # Get valid reference values
        valid_references = set(right_df[right_column].dropna().unique())

        # Check each row
        violating_rows = []
        valid_rows = []

        for idx, value in enumerate(left_df[left_column]):
            if pd.isna(value) or value is None:
                violating_rows.append(idx)
            elif value in valid_references:
                valid_rows.append(idx)
            else:
                violating_rows.append(idx)

        return violating_rows, valid_rows


class RuleFactory:
    """Factory class for creating rule instances from configuration."""

    _rule_classes = {
        'Referential Integrity': ReferentialIntegrityRule,
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
    """Main function to demonstrate the complete rule validation system with result saving."""

    # Load the datasets
    print("Loading datasets...")
    dataset = DataLoader.load_datasets()

    # Print dataset info
    print("\nDataset Summary:")
    print("-" * 50)
    for table_name, df in dataset.items():
        print(f"{table_name}: {len(df)} rows, {len(df.columns)} columns")

    # Create output directory
    output_dir = 'validation_results'
    os.makedirs(output_dir, exist_ok=True)

    # Define all rules from the JSON
    rules_json = [
        {
            "rule_type": "Referential Integrity",
            "rule": "Every responsabili.Codice_IPA must refer to an existing enti.Codice_IPA.",
            "explanation": "Every record in the responsabili_transizione_digitale table must reference a valid entity listed in the enti table via Codice_IPA, ensuring data integrity across these related tables.",
            "columns": [
                {"table": "responsabili", "name": "Codice_IPA"},
                {"table": "enti", "name": "Codice_IPA"}
            ],
            "join_pairs": [
                {
                    "left": {"table": "responsabili", "name": "Codice_IPA"},
                    "right": {"table": "enti", "name": "Codice_IPA"}
                }
            ]
        }
    ]

    # Process each rule
    print("\n" + "=" * 70)
    print("RULE VALIDATION RESULTS")
    print("=" * 70)

    for i, rule_config in enumerate(rules_json):
        print(f"\nRule {i + 1}: {rule_config['rule']}")
        print("-" * 70)

        try:
            # Create rule instance
            rule = RuleFactory.create_rule(rule_config)

            # Validate
            violating_rows, valid_rows = rule.validate(dataset)

            # Get report
            report = rule.get_violation_report(dataset)

            # Print results
            print(f"Total rows checked: {report['total_rows']}")
            print(f"Valid rows: {report['valid_rows']}")
            print(f"Violating rows: {report['violating_rows']}")
            print(f"Violation rate: {report['violation_rate']:.2f}%")

            # Show sample violations if any
            if violating_rows and len(violating_rows) > 0:
                join_pair = rule_config['join_pairs'][0]
                left_table = join_pair['left']['table']
                left_column = join_pair['left']['name']

                print(f"\nSample violations (first 5):")
                df = dataset[left_table]
                sample_violations = df.iloc[violating_rows[:5]]
                print(sample_violations[[left_column]].to_string())

            # Save validation results
            rule.save_validation_results(dataset, output_dir)

        except Exception as e:
            print(f"Error validating rule: {e}")

    # Overall summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total rules processed: {len(rules_json)}")


if __name__ == "__main__":
    main()