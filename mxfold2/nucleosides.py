import dataclasses
from typing import Dict, Optional, Set, Tuple


@dataclasses.dataclass
class Bases:
    code: str
    origin: str
    pairedwith: str
    smiles: str
    description: str
    custom_pairing: bool = False  # True if this base has custom pairing rules (not inherited from origin)

supported_nucleosides = {
    'A': Bases('A', 'A', 'U', 'Nc1ncnc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', 'adenosine'),
    'C': Bases('C', 'C', 'G', 'Nc1nc(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)cc1', 'cytidine'),
    'G': Bases('G', 'G', 'CU', 'Nc1[nH]c(=O)c2nc[n]([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)c2n1', 'guanosine'),
    'U': Bases('U', 'U', 'AG', 'OC[C@H]1O[C@@H]([n]2ccc(=O)[nH]c2=O)[C@H](O)[C@@H]1O', 'uridine'),
    'I': Bases('I', 'A', 'UC', 'OC[C@@H]1[C@@H](O)[C@@H](O)[C@H]([n]2c3nc[nH]c(=O)c3nc2)O1', 'inosine'),
    'P': Bases('P', 'U', 'A', 'OC[C@H]1O[C@@H](c2c[nH]c(=O)[nH]c2=O)[C@H](O)[C@@H]1O', 'pseudouridine'),
    '6': Bases('6', 'A', 'U', 'CNc1ncnc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', 'N6-methyladenosine'),
    '?': Bases('?', 'C', 'G', 'Cc1c(N)nc(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c1', '5-methylcytidine'),
#    '1': Bases('1', 'A', 'U', 'C[n]1c(=N)c2c([n]([C@H]3[C@H](O)[C@H](O)[C@@H](CO)O3)cn2)nc1', '1-methyladenosine'),
    '1': Bases('1', 'U', 'AG', 'C[n]1c(=O)[nH]c(=O)c([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c1', '1-methylpseudouridine'),
#    ':': Bases(':', 'A', 'U', 'CO[C@H]1[C@H]([n]2cnc3c2ncnc3N)O[C@H](CO)[C@H]1O', '2\'-O-methyladenosine'),
#    'B': Bases('B', 'C', 'G', 'CO[C@H]1[C@H]([n]2ccc(N)nc2=O)O[C@H](CO)[C@H]1O', '2\'-O-methylcytidine'),
#    '#': Bases('#', 'G', 'CU', 'CO[C@H]1[C@H]([n]2cnc3c2nc(N)[nH]c3=O)O[C@H](CO)[C@H]1O', '2\'-O-methylguanosine'),
#    'J': Bases('J', 'U', 'AG', 'CO[C@H]1[C@H]([n]2ccc(=O)[nH]c2=O)O[C@H](CO)[C@H]1O', '2\'-O-methyluridine'),
    '7': Bases('7', 'A', 'U', 'Nc1ncnc2c1ccn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', '7-deaza-adenonsine (7DA)'),
    'D': Bases('D', 'U', 'A', 'OC[C@H]1O[C@@H](N2CCC(=O)NC2=O)[C@H](O)[C@@H]1O', 'dihydrouridine')
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


def register_nucleoside(
    code: str,
    origin: str,
    pairedwith: str,
    smiles: str = "",
    description: str = "",
    custom_pairing: bool = False,
) -> Bases:
    """Register a new nucleoside (modified base) dynamically.

    Args:
        code: Single character code for the nucleoside (can be Unicode)
        origin: The canonical/parent base (A, C, G, or U)
        pairedwith: String of bases this nucleoside can pair with
        smiles: SMILES representation of the structure (optional)
        description: Human-readable description (optional)
        custom_pairing: True if pairing rules differ from origin base

    Returns:
        The newly registered Bases object
    """
    base = Bases(
        code=code,
        origin=origin,
        pairedwith=pairedwith,
        smiles=smiles,
        description=description,
        custom_pairing=custom_pairing,
    )
    supported_nucleosides[code] = base
    return base


def generate_pairing_rules(
    include_standard: bool = True,
) -> Dict[Tuple[str, str], bool]:
    """Generate pairing rules dictionary from supported nucleosides.

    This creates a dictionary suitable for passing to the C++ fold interface.
    Format: {('base1', 'base2'): True, ...}

    Args:
        include_standard: Whether to include standard base pairs (A-U, G-C, G-U)

    Returns:
        Dictionary mapping (base1, base2) tuples to bool indicating if pairing is allowed
    """
    rules: Dict[Tuple[str, str], bool] = {}

    if include_standard:
        # Standard Watson-Crick and wobble pairs
        standard_pairs = [
            ('a', 'u'), ('u', 'a'),
            ('A', 'U'), ('U', 'A'),
            ('g', 'c'), ('c', 'g'),
            ('G', 'C'), ('C', 'G'),
            ('g', 'u'), ('u', 'g'),
            ('G', 'U'), ('U', 'G'),
        ]
        for pair in standard_pairs:
            rules[pair] = True

    # Add pairing rules for all supported nucleosides
    for code, base in supported_nucleosides.items():
        for partner in base.pairedwith:
            # Add both cases (uppercase and lowercase)
            rules[(code.lower(), partner.lower())] = True
            rules[(partner.lower(), code.lower())] = True
            rules[(code.upper(), partner.upper())] = True
            rules[(partner.upper(), code.upper())] = True
            # Mixed case
            rules[(code.lower(), partner.upper())] = True
            rules[(partner.upper(), code.lower())] = True
            rules[(code.upper(), partner.lower())] = True
            rules[(partner.lower(), code.upper())] = True

    return rules


def generate_allowed_pairs_string() -> str:
    """Generate allowed_pairs string in the traditional format.

    Returns a string where consecutive character pairs represent allowed pairings.
    Example: "aucggua6" means a-u, c-g, g-u, a-6 are allowed pairs.

    Returns:
        String of allowed pair characters
    """
    pairs_set: Set[Tuple[str, str]] = set()

    for code, base in supported_nucleosides.items():
        for partner in base.pairedwith:
            # Normalize to lowercase and ensure consistent ordering
            pair = tuple(sorted([code.lower(), partner.lower()]))
            pairs_set.add(pair)

    # Convert to string format
    result = ""
    for p1, p2 in sorted(pairs_set):
        result += p1 + p2

    return result


def get_canonical_base(code: str) -> str:
    """Get the canonical (parent) base for a given nucleoside code.

    Args:
        code: Nucleoside code

    Returns:
        The canonical base (A, C, G, or U), or the code itself if unknown
    """
    if code in supported_nucleosides:
        return supported_nucleosides[code].origin
    # For standard bases, return themselves
    if code.upper() in {'A', 'C', 'G', 'U', 'T'}:
        return 'U' if code.upper() == 'T' else code.upper()
    return code


def get_pairing_partners(code: str) -> str:
    """Get the bases that can pair with the given nucleoside.

    Args:
        code: Nucleoside code

    Returns:
        String of bases that can pair with this nucleoside
    """
    if code in supported_nucleosides:
        return supported_nucleosides[code].pairedwith
    # For standard bases, return standard pairing partners
    standard_pairs = {
        'A': 'U', 'a': 'u',
        'U': 'AG', 'u': 'ag',
        'G': 'CU', 'g': 'cu',
        'C': 'G', 'c': 'g',
        'T': 'A', 't': 'a',
    }
    return standard_pairs.get(code, "")


def generate_nucleoside_info_for_cpp() -> Dict[str, Dict[str, str]]:
    """Generate nucleoside information for C++ (origin, pairedwith).

    This creates a dictionary suitable for passing to the C++ fold interface
    for dynamic base pair type determination.

    Returns:
        Dictionary mapping nucleoside codes to their properties:
        {code: {'origin': origin_base, 'pairedwith': pairing_partners}, ...}
    """
    result: Dict[str, Dict[str, str]] = {}
    for code, base in supported_nucleosides.items():
        result[code] = {
            'origin': base.origin,
            'pairedwith': base.pairedwith,
        }
    return result