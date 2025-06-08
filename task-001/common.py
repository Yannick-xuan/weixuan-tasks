"""
Common code shared across rule validation modules.
Contains base classes, factories, and utilities used by all rule types.
"""
import pandas as pd
from typing import Dict, List, Tuple, Any
from abc import ABC, abstractmethod
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


class RuleFactory:
    """Factory class for creating rule instances from configuration."""

    _rule_classes = {}

    @classmethod
    def register_rule_type(cls, rule_type: str, rule_class: type):
        """Register a new rule type with its implementation class."""
        cls._rule_classes[rule_type] = rule_class

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