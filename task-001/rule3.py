"""
Complete implementation and demo for Conditional Reference Existence rules
"""
import pandas as pd
from typing import Dict, List, Tuple, Any
import re
from abc import ABC, abstractmethod
import os


# Include all necessary classes (in practice, these would be imported)
class Rule(ABC):
    """Abstract base class for all rule types."""

    def __init__(self, rule_config: Dict[str, Any]):
        self.rule_type = rule_config.get('rule_type')
        self.rule = rule_config.get('rule')
        self.explanation = rule_config.get('explanation')
        self.columns = rule_config.get('columns', [])
        self.join_pairs = rule_config.get('join_pairs', [])
        self._validate_config()

    @abstractmethod
    def _validate_config(self):
        pass

    @abstractmethod
    def validate(self, dataset: Dict[str, pd.DataFrame]) -> Tuple[List[int], List[int]]:
        pass

    def get_violation_report(self, dataset: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
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

        if 'contains' not in self.rule and 'CONTAINS' not in self.rule:
            raise ValueError("Conditional Reference rule must contain 'contains' keyword in the rule text")

    def _parse_condition(self) -> Dict[str, Any]:
        """Parse the rule text to extract the condition and reference requirement."""
        # Pattern: IF table.column contains 'text' THEN table.column must exist in table.column
        pattern = r"IF\s+(\w+)\.(\w+)\s+contains\s+'([^']+)'\s+THEN\s+(\w+)\.(\w+)\s+must\s+exist\s+in\s+(\w+)\.(\w+)"
        match = re.search(pattern, self.rule, re.IGNORECASE)

        if not match:
            raise ValueError(f"Could not parse rule: {self.rule}")

        return {
            'condition_table': match.group(1),
            'condition_column': match.group(2),
            'search_text': match.group(3),
            'check_table': match.group(4),
            'check_column': match.group(5),
            'reference_table': match.group(6),
            'reference_column': match.group(7)
        }

    def validate(self, dataset: Dict[str, pd.DataFrame]) -> Tuple[List[int], List[int]]:
        """
        Validate conditional reference rule on the dataset.

        This rule checks: IF table1.col1 contains 'text' THEN table1.col2 must exist in table2.col3
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
        search_text = condition['search_text'].lower()  # Case-insensitive search

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

            # Check if condition is met (contains the search text)
            if pd.notna(condition_value) and search_text in str(condition_value).lower():
                # Condition is met - must check reference
                if pd.isna(check_value) or check_value not in valid_references:
                    violating_rows.append(idx)
                else:
                    valid_rows.append(idx)
            else:
                # Condition not met - automatically valid
                valid_rows.append(idx)

        return violating_rows, valid_rows


def main():
    """Main function to demonstrate Conditional Reference Existence rules"""

    print("="*70)
    print("Conditional Reference Existence VALIDATION")
    print("="*70)

    # Load the real datasets
    print("\nLoading datasets...")
    dataset = DataLoader.load_datasets()

    # Print dataset info
    print("\nDataset Summary:")
    print("-" * 50)
    for table_name, df in dataset.items():
        print(f"{table_name}: {len(df)} rows, {len(df.columns)} columns")

    # Quick preview of the data
    if 'uo' in dataset and 'enti' in dataset:
        uo_data = dataset['uo']
        enti_data = dataset['enti']

        print("\nSample of UO (Organizational Units) descriptions:")
        # Find rows containing 'transizione' or 'digitale' for preview
        if 'Descrizione_uo' in uo_data.columns:
            mask = uo_data['Descrizione_uo'].str.contains('transizione|digitale', case=False, na=False)
            sample_rows = uo_data[mask].head(5)
            if not sample_rows.empty:
                print(sample_rows[['Codice_uni_uo', 'Codice_IPA', 'Descrizione_uo']].to_string())
            else:
                print("No rows containing 'transizione' or 'digitale' found in descriptions")

        print("\nTotal unique Codice_IPA in enti table:", enti_data['Codice_IPA'].nunique())
        print("Total unique Codice_IPA in uo table:", uo_data['Codice_IPA'].nunique())

    # Define the rule
    rule_config = {
        "rule_type": "Conditional Reference Existence",
        "rule": "IF uo.Descrizione_uo contains 'transizione al digitale' THEN uo.Codice_IPA must exist in enti.Codice_IPA",
        "explanation": "When an organizational unit is designated as a digital transition office, it must reference an existing entity in the Entities table via IPA code. This ensures digital transition units are officially registered entities.",
        "columns": [
            {"table": "uo", "name": "Descrizione_uo"},
            {"table": "enti", "name": "Codice_IPA"}
        ],
        "join_pairs": [
            {
                "left": {"table": "uo", "name": "Codice_IPA"},
                "right": {"table": "enti", "name": "Codice_IPA"}
            }
        ]
    }

    print("\n" + "-"*70)
    print("VALIDATION RESULTS")
    print("-"*70)
    print(f"Rule: {rule_config['rule']}")
    print(f"Explanation: {rule_config['explanation']}")

    try:
        # Create and validate the rule
        rule = ConditionalReferenceExistence(rule_config)
        violating_rows, valid_rows = rule.validate(dataset)
        report = rule.get_violation_report(dataset)

        print(f"\nTotal rows checked: {report['total_rows']}")
        print(f"Valid rows: {report['valid_rows']}")
        print(f"Violating rows: {report['violating_rows']}")
        print(f"Violation rate: {report['violation_rate']:.2f}%")

        # Show details
        print("\n" + "-"*70)
        print("DETAILED ANALYSIS")
        print("-"*70)

        if 'uo' in dataset:
            uo_data = dataset['uo']

            # Find all rows containing 'transizione al digitale'
            matching_indices = []
            print("\nRows containing 'transizione al digitale':")
            for idx in range(len(uo_data)):
                desc = uo_data.iloc[idx]['Descrizione_uo']
                if pd.notna(desc) and 'transizione al digitale' in str(desc).lower():
                    matching_indices.append(idx)
                    ipa = uo_data.iloc[idx]['Codice_IPA']
                    status = "VIOLATION" if idx in violating_rows else "VALID"
                    print(f"  Row {idx}: {desc[:60]}...")
                    print(f"    Codice_IPA: {ipa} - {status}")

            if not matching_indices:
                print("  No rows found containing 'transizione al digitale'")

            print(f"\nTotal rows with 'transizione al digitale': {len(matching_indices)}")

            # Show sample violations
            print("\nSample violating records (first 10):")
            if violating_rows:
                sample_violations = uo_data.iloc[violating_rows[:10]]
                cols_to_show = ['Codice_uni_uo', 'Codice_IPA', 'Descrizione_uo']
                available_cols = [col for col in cols_to_show if col in sample_violations.columns]
                print(sample_violations[available_cols].to_string(index=True))
            else:
                print("  No violations found")

        # ========== 新增：输出 summary、valid、violation 明细到 validation_results 目录 ==========
        output_dir = 'validation_results'
        os.makedirs(output_dir, exist_ok=True)

        # # summary
        # summary_data = [{
        #     'Rule_Type': report['rule_type'],
        #     'Rule_Description': rule_config['rule'],
        #     'Total_Rows': report['total_rows'],
        #     'Valid_Rows': report['valid_rows'],
        #     'Violating_Rows': report['violating_rows'],
        #     'Violation_Rate_%': round(report['violation_rate'], 2)
        # }]
        # summary_df = pd.DataFrame(summary_data)
        # summary_file = os.path.join(output_dir, 'ConditionalReference_summary.csv')
        # summary_df.to_csv(summary_file, index=False)
        # print(f"Results exported to: {summary_file}")

        # valid/violation
        join_pair = rule_config['join_pairs'][0]
        left_table = join_pair['left']['table']
        if left_table in dataset:
            left_df = dataset[left_table]
            # valid
            if report['valid_indices']:
                valid_rows_df = left_df.iloc[report['valid_indices']].copy()
                print(report['valid_indices'])
                valid_rows_df['validation_status'] = 'valid'
                valid_file = os.path.join(output_dir, f'ConditionalReference_valid_rule_1.csv')
                valid_rows_df.to_csv(valid_file, index=False)
                print(f"Valid rows saved to: {valid_file}")
            # violation
            if report['violating_indices']:
                print(report['violating_indices'])
                violation_rows_df = left_df.iloc[report['violating_indices']].copy()
                violation_rows_df['validation_status'] = 'violation'
                violation_file = os.path.join(output_dir, f'ConditionalReference_violation_rule_1.csv')
                violation_rows_df.to_csv(violation_file, index=False)
                print(f"Violation rows saved to: {violation_file}")
        # ========== END ==========

    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()


    # Test with different conditions
    print("\n" + "="*70)
    print("TESTING OTHER CONDITIONS")
    print("="*70)


# Add DataLoader class if not already present
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


if __name__ == "__main__":
    main()