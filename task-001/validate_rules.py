"""
CLI script for batch processing data validation rules.
"""
import argparse
import os
import sys
from typing import Dict, Any, List
import pandas as pd
from common import DataLoader, RuleLoader, RuleFactory
from rule1 import ReferentialIntegrityRule
from rule2 import CrossTableConsistencyRule
from rule3 import ConditionalReferenceExistence
from rule4 import ConditionalReferenceWithLabelRestrictionRule

def setup_rule_types():
    """Register all available rule types with the factory."""
    RuleFactory.register_rule_type('Referential Integrity', ReferentialIntegrityRule)
    RuleFactory.register_rule_type('Cross-Table Consistency', CrossTableConsistencyRule)
    RuleFactory.register_rule_type('Conditional Reference Existence', ConditionalReferenceExistence)
    RuleFactory.register_rule_type('Conditional Reference with Label Restriction', ConditionalReferenceWithLabelRestrictionRule)

def process_rules(dataset: Dict[str, pd.DataFrame], rules: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
    """
    Process all rules against the dataset.

    Parameters:
    -----------
    dataset : Dict[str, pd.DataFrame]
        Dictionary mapping table names to DataFrames
    rules : List[Dict[str, Any]]
        List of rule configurations to process
    output_dir : str
        Directory to save validation results

    Returns:
    --------
    List[Dict[str, Any]]
        List of validation reports for each rule
    """
    all_results = []
    
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    # First, validate that all required tables are available
    required_tables = set()
    for rule_config in rules:
        # Get tables from join pairs
        if 'join_pairs' in rule_config:
            for join_pair in rule_config['join_pairs']:
                if 'left' in join_pair:
                    required_tables.add(join_pair['left']['table'])
                if 'right' in join_pair:
                    required_tables.add(join_pair['right']['table'])
        # Get tables from columns
        if 'columns' in rule_config:
            for column in rule_config['columns']:
                if 'table' in column:
                    required_tables.add(column['table'])

    # Check for missing tables
    missing_tables = required_tables - set(dataset.keys())
    if missing_tables:
        print("\nWARNING: The following required tables are missing:")
        for table in missing_tables:
            print(f"  - {table}")
        print("\nAvailable tables:")
        for table in dataset.keys():
            print(f"  - {table}")
        print("\nSome rules may fail due to missing tables.")

    for i, rule_config in enumerate(rules, 1):
        print(f"\nRule {i}/{len(rules)}: {rule_config['rule']}")
        print("-" * 70)

        try:
            # Create rule instance
            rule = RuleFactory.create_rule(rule_config)

            # Check if all required tables for this rule are available
            rule_tables = set()
            for join_pair in rule_config.get('join_pairs', []):
                if 'left' in join_pair:
                    rule_tables.add(join_pair['left']['table'])
                if 'right' in join_pair:
                    rule_tables.add(join_pair['right']['table'])

            missing_rule_tables = rule_tables - set(dataset.keys())
            if missing_rule_tables:
                print(f"Skipping rule due to missing tables: {', '.join(missing_rule_tables)}")
                continue

            # Validate
            violating_rows, valid_rows = rule.validate(dataset)
            report = rule.get_violation_report(dataset)
            all_results.append(report)

            # Print results
            print(f"Total rows checked: {report['total_rows']}")
            print(f"Valid rows: {report['valid_rows']}")
            print(f"Violating rows: {report['violating_rows']}")
            print(f"Violation rate: {report['violation_rate']:.2f}%")

            # Save validation results
            if output_dir:
                rule.save_validation_results(dataset, output_dir)

        except Exception as e:
            print(f"Error validating rule: {str(e)}")
            if 'Table' in str(e) and 'not found' in str(e):
                print("This error occurred because a required table was not found in the dataset.")
                print("Please ensure all required tables are available.")
            import traceback
            traceback.print_exc()

    return all_results

def print_summary(all_results: List[Dict[str, Any]]):
    """Print summary statistics for all processed rules."""
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
        for i, result in enumerate(sorted_results[:3], 1):
            print(f"  {i}. {result['rule'][:60]}...")
            print(f"     Violation rate: {result['violation_rate']:.2f}%")

def main():
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(description='Validate data rules in batch mode.')
    parser.add_argument('--data-dir', default='.',
                      help='Directory containing the Excel data files (default: current directory)')
    parser.add_argument('--json-dir', default='json',
                      help='Directory containing the JSON rule configurations (default: json)')
    parser.add_argument('--output-dir', default='validation_results',
                      help='Directory to save validation results (default: validation_results)')
    parser.add_argument('--rule-type', choices=['all', 'referential', 'consistency', 'conditional', 'label-restriction'],
                      default='all',
                      help='Type of rules to process (default: all)')
    
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Register rule types
    setup_rule_types()

    # Load datasets
    print("\nLoading datasets from:", args.data_dir)
    dataset = DataLoader.load_datasets(args.data_dir)
    if not dataset:
        print("Error: No datasets loaded. Please check the data directory.")
        sys.exit(1)

    # Print dataset info
    print("\nDataset Summary:")
    print("-" * 50)
    for table_name, df in dataset.items():
        print(f"{table_name}: {len(df)} rows, {len(df.columns)} columns")

    # Load rules
    print("\nLoading rules from:", args.json_dir)
    try:
        all_rules = RuleLoader.load_rules(args.json_dir)
        if not all_rules:
            print("Error: No rules loaded. Please check the JSON directory.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading rules: {e}")
        sys.exit(1)

    # Filter rules based on type if specified
    if args.rule_type != 'all':
        rule_type_mapping = {
            'referential': 'Referential Integrity',
            'consistency': 'Cross-Table Consistency',
            'conditional': 'Conditional Reference Existence',
            'label-restriction': 'Conditional Reference with Label Restriction'
        }
        selected_type = rule_type_mapping.get(args.rule_type)
        all_rules = [r for r in all_rules if r['rule_type'] == selected_type]
        if not all_rules:
            print(f"No rules found for type: {args.rule_type}")
            sys.exit(1)

    # Process rules
    print(f"\nProcessing {len(all_rules)} rules...")
    all_results = process_rules(dataset, all_rules, args.output_dir)

    # Print summary
    print_summary(all_results)

if __name__ == "__main__":
    main() 