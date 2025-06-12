"""
Unit tests for data validation rules.

This module tests all four rule types:
- Referential Integrity
- Cross-Table Consistency
- Conditional Reference Existence
- Conditional Reference with Label Restriction
"""
import unittest
import pandas as pd
import os
import json
from common import DataLoader, RuleLoader, RuleFactory
from rule1 import ReferentialIntegrityRule
from rule2 import CrossTableConsistencyRule
from rule3 import ConditionalReferenceExistence
from rule4 import ConditionalReferenceWithLabelRestrictionRule

RuleFactory.register_rule_type('Referential Integrity', ReferentialIntegrityRule)
RuleFactory.register_rule_type('Cross-Table Consistency', CrossTableConsistencyRule)
RuleFactory.register_rule_type('Conditional Reference Existence', ConditionalReferenceExistence)
RuleFactory.register_rule_type('Conditional Reference with Label Restriction',
                               ConditionalReferenceWithLabelRestrictionRule)


class TestValidationRules(unittest.TestCase):
    """Test suite for all validation rule types."""

    @classmethod
    def setUpClass(cls):
        """
        Load datasets once for all tests.

        This method runs once before all tests in the class.
        It loads all Excel files from the current directory.

        Raises:
            ValueError: If no datasets can be loaded.
        """
        cls.data_loader = DataLoader()
        cls.datasets = cls.data_loader.load_datasets(data_dir='.')
        if not cls.datasets:
            raise ValueError("No datasets loaded. Please check the data directory.")

    def _run_rule_test(self, rule_config_file: str, expected_total: int, expected_valid: int,
                       expected_violating: int, expected_violation_rate: float, rule_display_name: str):
        """
        Helper function to run a single rule test.

        Parameters:
        -----------
        rule_config_file : str
            Path to the JSON file containing the rule configuration
        expected_total : int
            Expected total number of rows checked
        expected_valid : int
            Expected number of valid rows
        expected_violating : int
            Expected number of violating rows
        expected_violation_rate : float
            Expected violation rate as a percentage
        rule_display_name : str
            Display name for the rule in test output

        Raises:
        -------
        ValueError
            If the rule configuration cannot be loaded from the file
        AssertionError
            If any of the actual values don't match expected values
        """
        # Load the specific rule configuration from the JSON file
        with open(rule_config_file, 'r', encoding='utf-8') as f:
            specific_rule_config_list = json.load(f)

        # Extract the rule configuration
        # Handle both single rule object and list of rules
        if not isinstance(specific_rule_config_list, list):
            current_rule_config = specific_rule_config_list
        elif specific_rule_config_list:
            current_rule_config = specific_rule_config_list[0]  # Take the first rule from the list
        else:
            current_rule_config = None

        if current_rule_config is None:
            raise ValueError(f"Test rule not found in {rule_config_file}")

        # Create rule instance using factory pattern
        rule_instance = RuleFactory.create_rule(current_rule_config)

        violating_rows, valid_rows = rule_instance.validate(self.datasets)
        report = rule_instance.get_violation_report(self.datasets)

        print(f"\n{rule_display_name}: {rule_instance.rule}")
        print("-" * 70)
        print(f"Total rows checked: {report['total_rows']}")
        print(f"Valid rows: {report['valid_rows']}")
        print(f"Violating rows: {report['violating_rows']}")
        print(f"Violation rate: {report['violation_rate']:.2f}%")

        self.assertEqual(report['total_rows'], expected_total)
        self.assertEqual(report['valid_rows'], expected_valid)
        self.assertEqual(report['violating_rows'], expected_violating)
        self.assertAlmostEqual(report['violation_rate'], expected_violation_rate, places=2)

    def test_rule_1_referential_integrity(self):
        """
        Test Referential Integrity rule.

        Rule: Every responsabili.Codice_IPA must refer to an existing enti.Codice_IPA.

        Expected results based on test data:
        - Total rows: 20,518
        - Valid rows: 20,518
        - Violating rows: 0
        - Violation rate: 0.00%
        """
        self._run_rule_test(
            rule_config_file='json/test/test1.json',
            expected_total=20518,
            expected_valid=20518,
            expected_violating=0,
            expected_violation_rate=0.00,
            rule_display_name="Rule 1/4 (Referential Integrity)"
        )

    def test_rule_2_cross_table_consistency(self):
        """
        Test Cross-Table Consistency rule.

        Rule: IF uo.Codice_uni_uo = responsabili.Codice_uni_uo 
              THEN uo.Descrizione_uo = responsabili.Descrizione_uo

        Expected results based on test data:
        - Total rows: 118,274
        - Valid rows: 118,274
        - Violating rows: 0
        - Violation rate: 0.00%
        """
        self._run_rule_test(
            rule_config_file='json/test/test2.json',
            expected_total=118274,
            expected_valid=118274,
            expected_violating=0,
            expected_violation_rate=0.00,
            rule_display_name="Rule 2/4 (Cross-Table Consistency)"
        )

    def test_rule_3_conditional_reference_existence(self):

        self._run_rule_test(
            rule_config_file='json/test/test3.json',
            expected_total=118274,
            expected_valid=118274,
            expected_violating=0,
            expected_violation_rate=0.00,
            rule_display_name="Rule 3/4 (Conditional Reference Existence)"
        )

    def test_rule_4_conditional_label_restriction(self):
        """
        Test Conditional Reference with Label Restriction rule.

        Rule: IF responsabili.Codice_uni_uo = uo.Codice_uni_uo 
              THEN uo.Descrizione_uo contains 'transizione al digitale'

        Expected results based on test data:
        - Total rows: 20,518
        - Valid rows: 20,518
        - Violating rows: 0
        - Violation rate: 0.00%
        """
        self._run_rule_test(
            rule_config_file='json/test/test4.json',
            expected_total=20518,
            expected_valid=20518,
            expected_violating=0,
            expected_violation_rate=0.00,
            rule_display_name="Rule 4/4 (Conditional Reference with Label Restriction)"
        )

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)