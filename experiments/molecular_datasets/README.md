# Molecular Datasets for Q-Genesis Experiments

This directory contains molecular geometry files and datasets for
benchmarking and testing Q-Genesis.

## Directory Structure

```
molecular_datasets/
├── small_molecules/      # H2, LiH, BeH2
├── medium_molecules/     # H2O, NH3, CH4
├── large_molecules/      # C2H4, C2H6, small aromatics
└── reaction_paths/       # Transition state structures
```

## File Formats

- `.xyz` - XYZ coordinate files (Angstrom)
- `.mol` - MDL mol files with connectivity
- `.json` - Metadata and reference energies

## Reference Energies

All reference energies are computed at the FCI/cc-pVDZ level
unless otherwise noted.

## Adding New Molecules

1. Create XYZ file following standard format:
   ```
   <n_atoms>
   <comment line with energy>
   <symbol> <x> <y> <z>
   ...
   ```

2. Add reference energy to `reference_energies.json`

3. Run validation: `python validate_dataset.py`
