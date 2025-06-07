# Rule Validation System Documentation

## 1. Implementation Overview

### 1.1 Rule Abstraction

All rules inherit from the abstract base class `Rule`, which requires:

- `_validate_config()`: Validate the rule configuration
- `validate(dataset)`: Return indices of violating and valid rows
- `get_violation_report(dataset)`: Return detailed statistics
- `save_validation_results(dataset, output_dir)`: Save details to CSV

### 1.2 Rule Types

- **rule1.py**: Referential Integrity
- **rule2.py**: Cross-Table Consistency
- **rule3.py**: Conditional Reference Existence
- **rule4.py**: Conditional Reference with Label Restriction

### 1.3 General Export Logic

- Automatically detects the "left table" and exports original data + `validation_status`
- Standardized file naming for batch processing and traceability
- All outputs are saved in the `validation_results` directory

---

## 2. Input Format Specification

### 2.1 Dataset

- Type: `Dict[str, pd.DataFrame]`
- Structure: Keys are table names, values are DataFrames
- Example:

```python
dataset = {
    'uo': pd.DataFrame(...),
    'enti': pd.DataFrame(...),
    'responsabili': pd.DataFrame(...)
}
```

### 2.2 Rule Instance

- Type: `Dict[str, Any]`
- Main fields:
    - `rule_type`: Rule type (e.g., "Referential Integrity")
    - `rule`: Rule description (in natural language or pseudo-code)
    - `explanation`: Rule explanation
    - `columns`: Involved tables and columns
    - `join_pairs`: Join relationships (e.g., foreign key, primary key)
- Example (Referential Integrity):

```python
{
    "rule_type": "Referential Integrity",
    "rule": "Every responsabili.Codice_IPA must refer to an existing enti.Codice_IPA.",
    "explanation": "...",
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
```

---

## 3. Output File Specification(and print row list)

Each validation outputs the following files to the `validation_results` directory:

- **Valid details**: All columns from the left table + `validation_status=valid`
- **Violation details**: All columns from the left table + `validation_status=violation`
- **Summary (optional)**: Rule statistics

Example file names:

- `Referential_Integrity_valid.csv`
- `Referential_Integrity_violating.csv`
- `CrossTableConsis_valid_rule_1.csv`
- `CrossTableConsis_violation_rule_1.csv`
- `ConditionalReferenceWithLabelRestriction_valid_rule_1.csv`
- `ConditionalReferenceWithLabelRestriction_violation_rule_1.csv`

---

If there is no such violation row file(now violation row print ), then there is no non-compliant data

## 4. Usage Example

### 4.1 Running a Single Rule Script

Assuming your Excel data files are in the current directory, simply run the corresponding script:

```Bash
python rule1.py
python rule2.py
python rule3.py
python rule4.py
```

### 4.2 Viewing Results

Check the `validation_results` directory for output CSV files.
Each file contains the original data and a `validation_status` column for easy analysis.

---

## 5. Main Assumptions & Edge Cases

- All tables are loaded as pandas DataFrames, and column names match the rule configuration
- Rule configuration must include necessary `join_pairs`, `columns`, etc.
- Missing values (NaN) are treated as violations (can be adjusted as needed)
- File name length is limited to avoid OS compatibility issues
- Each run processes one rule at a time; batch processing can be added via loops

---

## 6. Code Structure Overview

- `Rule`: Abstract base class defining the interface
- `ReferentialIntegrityRule`, `CrossTableConsistencyRule`, etc.: Concrete rule implementations
- `RuleFactory`: Rule factory for easy management and extension
- `DataLoader`: Utility for loading datasets
- `main()`: Script entry point for loading data, running validation, and exporting results

---

## 7. Example Function Documentation (from rule1.py)

```python
def validate(self, dataset: Dict[str, pd.DataFrame]) -> Tuple[List[int], List[int]]:
    '''
    Validate referential integrity rule.

    Parameters:
        dataset: Dict[str, pd.DataFrame]
            The dataset, with table names as keys and DataFrames as values

    Returns:
        Tuple[List[int], List[int]]
            - violating_rows: Indices of rows violating the rule
            - valid_rows: Indices of rows satisfying the rule

    Example usage:
        rule = ReferentialIntegrityRule(rule_config)
        violating_rows, valid_rows = rule.validate(dataset)

    Assumptions:
        - All tables and columns exist
        - Missing values are treated as violations
    '''
```

---