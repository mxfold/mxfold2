import dataclasses
from typing import Set

@dataclasses.dataclass
class Bases:
    code: str
    origin: str
    pairedwith: str
    smiles: str
    description: str

supported_nucleosides = {
    'A': Bases('A', 'A', 'U', 'Nc1ncnc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', 'adenosine'),
    'C': Bases('C', 'C', 'G', 'Nc1nc(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)cc1', 'cytidine'),
    'G': Bases('G', 'G', 'CU', 'Nc1[nH]c(=O)c2nc[n]([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)c2n1', 'guanosine'),
    'U': Bases('U', 'U', 'AG', 'OC[C@H]1O[C@@H]([n]2ccc(=O)[nH]c2=O)[C@H](O)[C@@H]1O', 'uridine'),
    'I': Bases('I', 'A', 'UCA', 'OC[C@@H]1[C@@H](O)[C@@H](O)[C@H]([n]2c3nc[nH]c(=O)c3nc2)O1', 'inosine'),
    'Y': Bases('Y', 'U', 'AGU', 'OC[C@H]1O[C@@H](c2c[nH]c(=O)[nH]c2=O)[C@H](O)[C@@H]1O', 'pseudouridine'),
    '6': Bases('6', 'A', 'U', 'CNc1ncnc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', 'N6-methyladenosine'),
    '5': Bases('5', 'C', 'G', 'Cc1c(N)nc(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c1', '5-methylcytidine'),
    '1': Bases('1', 'A', 'U', 'C[n]1c(=N)c2c([n]([C@H]3[C@H](O)[C@H](O)[C@@H](CO)O3)cn2)nc1', '1-methyladenosine'),
    'P': Bases('P', 'U', 'AGU', 'C[n]1c(=O)[nH]c(=O)c([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c1', '1-methylpseudouridine'),
    ':': Bases(':', 'A', 'U', 'CO[C@H]1[C@H]([n]2cnc3c2ncnc3N)O[C@H](CO)[C@H]1O', '2\'-O-methyladenosine'),
    'B': Bases('B', 'C', 'G', 'CO[C@H]1[C@H]([n]2ccc(N)nc2=O)O[C@H](CO)[C@H]1O', '2\'-O-methylcytidine'),
    '#': Bases('#', 'G', 'CU', 'CO[C@H]1[C@H]([n]2cnc3c2nc(N)[nH]c3=O)O[C@H](CO)[C@H]1O', '2\'-O-methylguanosine'),
    'J': Bases('J', 'U', 'AG', 'CO[C@H]1[C@H]([n]2ccc(=O)[nH]c2=O)O[C@H](CO)[C@H]1O', '2\'-O-methyluridine'),
}


# Standard bases (ACGU)
STANDARD_BASES: frozenset[str] = frozenset({'A', 'C', 'G', 'U', 'a', 'c', 'g', 'u'})


def is_modified_base(base: str) -> bool:
    """Determine whether a base is a modified base.

    Args:
        base: Single character code of the base

    Returns:
        True if modified base, False if standard base (ACGU)
    """
    return base not in STANDARD_BASES


def get_modified_positions(seq: str) -> Set[int]:
    """Return positions of modified bases in the sequence (1-indexed).

    mxfold2 uses 1-indexed positions, so returned positions start from 1.

    Args:
        seq: Nucleotide sequence

    Returns:
        Set of modified base positions (1-indexed)
    """
    return {i + 1 for i, base in enumerate(seq) if is_modified_base(base)}


def has_modified_in_range(seq: str, start: int, end: int) -> bool:
    """Determine whether there are modified bases in the range [start, end] (1-indexed).

    Args:
        seq: Nucleotide sequence
        start: Start position (1-indexed, inclusive)
        end: End position (1-indexed, inclusive)

    Returns:
        True if there are modified bases in the range
    """
    for i in range(max(1, start), min(len(seq) + 1, end + 1)):
        if is_modified_base(seq[i - 1]):  # seq is 0-indexed
            return True
    return False