import dataclasses
from typing import Dict, Optional, Set, Tuple


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
    'P': Bases('P', 'U', 'AGU', 'OC[C@H]1O[C@@H](c2c[nH]c(=O)[nH]c2=O)[C@H](O)[C@@H]1O', 'pseudouridine'),
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
STANDARD_BASES: frozenset[str] = frozenset({'A', 'C', 'G', 'U', 'T', 'a', 'c', 'g', 'u', 't'})

# Characters that should be uppercased during normalization
_STANDARD_LOWER: frozenset[str] = frozenset({'a', 'c', 'g', 'u', 't'})


def normalize_seq(seq: str) -> str:
    """Normalize a sequence by uppercasing only standard bases (a/c/g/u/t).

    All other characters (modified base codes, Unicode symbols, etc.)
    are preserved as-is. This is critical because some modified base
    codes are case-sensitive (e.g., 'B' vs 'b' in MODOMICS).

    Args:
        seq: Nucleotide sequence string

    Returns:
        Sequence with only standard bases uppercased
    """
    return ''.join(c.upper() if c in _STANDARD_LOWER else c for c in seq)


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
) -> Bases:
    """Register a new nucleoside (modified base) dynamically.

    Args:
        code: Single character code for the nucleoside (can be Unicode)
        origin: The canonical/parent base (A, C, G, or U)
        pairedwith: String of bases this nucleoside can pair with
        smiles: SMILES representation of the structure (optional)
        description: Human-readable description (optional)

    Returns:
        The newly registered Bases object
    """
    base = Bases(
        code=code,
        origin=origin,
        pairedwith=pairedwith,
        smiles=smiles,
        description=description,
    )
    active_nucleosides[code] = base
    return base


def generate_pairing_rules(
    include_standard: bool = True,
) -> Dict[Tuple[str, str], bool]:
    """Generate pairing rules dictionary from active nucleosides.

    This creates a dictionary suitable for passing to the C++ fold interface.
    Format: {('base1', 'base2'): True, ...}

    Codes are used as-is (case-sensitive) since some modified base codes
    in MODOMICS use case to distinguish different molecules (e.g., 'B' vs 'b').
    Standard bases are represented in uppercase (after normalize_seq).

    Args:
        include_standard: Whether to include standard base pairs (A-U, G-C, G-U)

    Returns:
        Dictionary mapping (base1, base2) tuples to bool indicating if pairing is allowed
    """
    rules: Dict[Tuple[str, str], bool] = {}

    if include_standard:
        # Standard Watson-Crick and wobble pairs (uppercase, since
        # standard bases are always normalized to uppercase by normalize_seq)
        standard_pairs = [
            ('A', 'U'), ('U', 'A'),
            ('G', 'C'), ('C', 'G'),
            ('G', 'U'), ('U', 'G'),
        ]
        for pair in standard_pairs:
            rules[pair] = True

    # Add pairing rules for all active nucleosides
    for code, base in active_nucleosides.items():
        for partner in base.pairedwith:
            # Use code as-is (case-sensitive) and partner as uppercase
            # (partners in Bases definitions are standard uppercase letters)
            rules[(code, partner)] = True
            rules[(partner, code)] = True

    return rules


def generate_allowed_pairs_string() -> str:
    """Generate allowed_pairs string in the traditional format.

    Returns a string where consecutive character pairs represent allowed pairings.
    Example: "AUCGGUА6" means A-U, C-G, G-U, A-6 are allowed pairs.

    Returns:
        String of allowed pair characters
    """
    pairs_set: Set[Tuple[str, str]] = set()

    for code, base in active_nucleosides.items():
        for partner in base.pairedwith:
            pair = tuple(sorted([code, partner]))
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
    if code in active_nucleosides:
        return active_nucleosides[code].origin
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
    if code in active_nucleosides:
        return active_nucleosides[code].pairedwith
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
    for code, base in active_nucleosides.items():
        result[code] = {
            'origin': base.origin,
            'pairedwith': base.pairedwith,
        }
    return result

# obtained from MODOMICS (https://genesilico.pl/modomics/modifications)
modomics_nucleosides: Dict[str, Bases] = {
    '!': Bases('!', 'U', 'AG', 'OC[C@@H]1[C@@H](O)[C@@H](O)[C@H]([n]2c(=O)[nH]c(=O)c(CNCC(=O)O)c2)O1', '5-carboxymethylaminomethyluridine'),
    'Ѣ': Bases('Ѣ', 'A', 'U', 'C[n]1c(=N)c2c([n]([C@H]3[C@H](O)[C@H](O)[C@@H](CO)O3)cn2)nc1', '1-methyladenosine'),
    '＃': Bases('＃', 'G', 'CU', 'CO[C@H]1[C@H]([n]2cnc3c2nc(N)[nH]c3=O)O[C@H](CO)[C@H]1O', "2'-O-methylguanosine"),
    '$': Bases('$', 'U', 'AG', 'OC(CNCc1c[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=S)[nH]c1=O)=O', '5-carboxymethylaminomethyl-2-thiouridine'),
    'ʤ': Bases('ʤ', 'C', 'G', 'Nc1nc(=S)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)cc1', '2-thiocytidine'),
    '&': Bases('&', 'U', 'AG', 'NC(Cc1c(=O)[nH]c(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c1)=O', '5-carbamoylmethyluridine'),
    'Щ': Bases('Щ', 'C', 'G', 'C[n+]1c(N)cc[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c1=O', '3-methylcytidine'),
    '(': Bases('(', 'G', 'CU', 'Nc1[nH]c(=O)c2c(c[n]([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)c2n1)C(N)=N', 'archaeosine'),
    ')': Bases(')', 'U', 'AG', 'CO[C@H]1[C@H]([n]2cc(CNCC(O)=O)c(=O)[nH]c2=O)O[C@H](CO)[C@H]1O', "5-carboxymethylaminomethyl-2'-O-methyluridine"),
    '*': Bases('*', 'A', 'U', 'CC(C)=CCNc1nc(SC)nc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', '2-methylthio-N6-isopentenyladenosine'),
    'Ч': Bases('Ч', 'A', 'U', 'CC(C)=CCNc1ncnc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', 'N6-isopentenyladenosine'),
    'ɮ': Bases('ɮ', 'U', 'AG', 'COC(C(c1c(=O)[nH]c(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c1)O)=O', '5-(carboxyhydroxymethyl)uridine methyl ester'),
    'ɿ': Bases('ɿ', 'A', 'U', 'Cc1nc2c(nc[n]2[C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c(N)n1', '2-methyladenosine'),
    '1': Bases('1', 'U', 'AG', 'COC(Cc1c(=O)[nH]c(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c1)=O', '5-methoxycarbonylmethyluridine'),
    '2': Bases('2', 'U', 'AG', 'O[C@H]1[C@H]([n]2ccc(=O)[nH]c2=S)O[C@H](CO)[C@H]1O', '2-thiouridine'),
    '3': Bases('3', 'U', 'AG', 'COC(Cc1c[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=S)[nH]c1=O)=O', '5-methoxycarbonylmethyl-2-thiouridine'),
    '4': Bases('4', 'U', 'AG', 'OC[C@H]1O[C@@H]([n]2ccc(=S)[nH]c2=O)[C@H](O)[C@@H]1O', '4-thiouridine'),
    '5': Bases('5', 'U', 'AG', 'COc1c(=O)[nH]c(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c1', '5-methoxyuridine'),
    '6': Bases('6', 'A', 'U', 'CC(O)C(N)C(NC(Nc1ncnc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O)=O)=O', 'N6-threonylcarbamoyladenosine'),
    '7': Bases('7', 'G', 'CU', '[C@@H]1([n]2c[n+](C)c3c2nc(nc3=O)N)O[C@H](CO)[C@@H](O)[C@H]1O', '7-methylguanosine'),
    '8': Bases('8', 'G', 'CU', 'Nc1nc(=O)c2c([n]([C@H]3[C@H](O)[C@H](O)[C@@H](CO)O3)cc2CN[C@@H]2[C@@H](O[C@H]3C(O)[C@H](O)[C@H](O)C(CO)O3)[C@@H](O)C=C2)n1', 'mannosyl-queuosine'),
    '9': Bases('9', 'G', 'CU', 'Nc1[nH]c(=O)c2c([n]([C@H]3[C@H](O)[C@H](O)[C@@H](CO)O3)cc2CN[C@@H]2[C@@H](O[C@H]3C(O)[C@H](O)[C@@H](O)C(CO)O3)[C@@H](O)C=C2)n1', 'galactosyl-queuosine'),
    'ʍ': Bases('ʍ', 'A', 'U', 'CO[C@H]1[C@H]([n]2cnc3c2ncnc3N)O[C@H](CO)[C@H]1O', "2'-O-methyladenosine"),
    'Ж': Bases('Ж', 'A', 'U', 'CNc1ncnc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', 'N6-methyladenosine'),
    '>': Bases('>', 'C', 'G', 'Nc1nc(=O)[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)cc1C=O', '5-formylcytidine'),
    '?': Bases('?', 'C', 'G', 'Cc1c(N)nc(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c1', '5-methylcytidine'),
    'A': Bases('A', 'A', 'U', 'Nc1ncnc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', 'adenosine'),
    'B': Bases('B', 'C', 'G', 'CO[C@H]1[C@H]([n]2ccc(N)nc2=O)O[C@H](CO)[C@H]1O', "2'-O-methylcytidine"),
    'C': Bases('C', 'C', 'G', 'Nc1nc(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)cc1', 'cytidine'),
    'D': Bases('D', 'U', 'AG', 'OC[C@H]1O[C@@H](N2CCC(=O)NC2=O)[C@H](O)[C@@H]1O', 'dihydrouridine'),
    'E': Bases('E', 'A', 'U', 'C[C@H]([C@@H](C(=O)O)NC(Nc1c2c([n]([C@H]3[C@H](O)[C@H](O)[C@@H](CO)O3)cn2)ncn1)=O)O', 'N6-methyl-N6-threonylcarbamoyladenosine'),
    'F': Bases('F', 'U', 'AG', 'Cc1c[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=S)[nH]c1=O', '5-methyl-2-thiouridine'),
    'G': Bases('G', 'G', 'CU', 'Nc1[nH]c(=O)c2nc[n]([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)c2n1', 'guanosine'),
    'I': Bases('I', 'A', 'UCA', 'OC[C@@H]1[C@@H](O)[C@@H](O)[C@H]([n]2c3nc[nH]c(=O)c3nc2)O1', 'inosine'),
    'J': Bases('J', 'U', 'AG', 'CO[C@H]1[C@H]([n]2ccc(=O)[nH]c2=O)O[C@H](CO)[C@H]1O', "2'-O-methyluridine"),
    'K': Bases('K', 'G', 'CU', 'CN1C(=O)c2[n]c[n]([C@H]3[C@H](O)[C@H](O)[C@@H](CO)O3)c2N=C1N', '1-methylguanosine'),
    'L': Bases('L', 'G', 'CU', 'CNc1[nH]c(=O)c2nc[n]([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)c2n1', 'N2-methylguanosine'),
    'M': Bases('M', 'C', 'G', 'CC(Nc1nc(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)cc1)=O', 'N4-acetylcytidine'),
    'O': Bases('O', 'A', 'U', 'C[n]1c(=O)c2c([n]([C@H]3[C@H](O)[C@H](O)[C@@H](CO)O3)cn2)nc1', '1-methylinosine'),
    'P': Bases('P', 'U', 'AGU', 'OC[C@H]1O[C@@H](c2c[nH]c(=O)[nH]c2=O)[C@H](O)[C@@H]1O', 'pseudouridine'),
    'Q': Bases('Q', 'G', 'CU', 'Nc1[nH]c(=O)c2c([n]([C@H]3[C@H](O)[C@H](O)[C@@H](CO)O3)cc2CN[C@@H]2[C@@H](O)[C@@H](O)C=C2)n1', 'queuosine'),
    'R': Bases('R', 'G', 'CU', 'CN(C)c1[nH]c(=O)c2nc[n]([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)c2n1', 'N2,N2-dimethylguanosine'),
    'S': Bases('S', 'U', 'AG', 'CNCc1c[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=S)[nH]c1=O', '5-methylaminomethyl-2-thiouridine'),
    'T': Bases('T', 'U', 'AG', 'Cc1c(=O)[nH]c(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c1', '5-methyluridine'),
    'U': Bases('U', 'U', 'AG', 'OC[C@H]1O[C@@H]([n]2ccc(=O)[nH]c2=O)[C@H](O)[C@@H]1O', 'uridine'),
    'V': Bases('V', 'U', 'AG', 'OC[C@@H]1[C@@H](O)[C@@H](O)[C@H]([n]2c(=O)[nH]c(=O)c(OCC(=O)O)c2)O1', 'uridine 5-oxyacetic acid'),
    'W': Bases('W', 'G', 'CU', 'C[n]1c2c(nc[n]2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[n]2c1nc(C)c2CC(OO)C(NC(OC)=O)C(OC)=O', 'peroxywybutosine'),
    'X': Bases('X', 'U', 'AG', 'NC(CC[n]1c(=O)[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)ccc1=O)C(O)=O', '3-(3-amino-3-carboxypropyl)uridine'),
    'Y': Bases('Y', 'G', 'CU', 'COC(NC(C(OC)=O)CCc1[n]2c([n](c3c(c2=O)nc[n]3[C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)C)nc1C)=O', 'wybutosine'),
    '[': Bases('[', 'A', 'U', 'CC(O)C(NC(Nc1nc(SC)nc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O)=O)C(O)=O', '2-methylthio-N6-threonylcarbamoyladenosine'),
    'Ħ': Bases('Ħ', 'U', 'AG', 'CO[C@H]1[C@H]([n]2c(=O)[nH]c(=O)c(C)c2)O[C@H](CO)[C@H]1O', "5,2'-O-dimethyluridine"),
    ']': Bases(']', 'U', 'AG', 'C[n]1c(=O)[nH]c(=O)c([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c1', '1-methylpseudouridine'),
    'Ỽ': Bases('Ỽ', 'A', 'U', 'CC(CO)=CCNc1ncnc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', 'N6-(cis-hydroxyisopentenyl)adenosine'),
    'b': Bases('b', 'U', 'AG', 'COC(C(O)c1c[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2OC)c(=O)[nH]c1=O)=O', "5-(carboxyhydroxymethyl)-2'-O-methyluridine methyl ester"),
    'e': Bases('e', 'A', 'U', 'CC(O)C1NC(=Nc2ncnc3c2nc[n]3[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)OC1=O', 'cyclic N6-threonylcarbamoyladenosine'),
    'f': Bases('f', 'U', 'AG', 'CC(C)=CCC/C(/C)=C/CSc1nc(=O)c(CNCC(O)=O)c[n]1[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', '5-carboxymethylaminomethyl-2-geranylthiouridine'),
    'h': Bases('h', 'U', 'AG', 'CC(C)=CCC/C(/C)=C/CSc1nc(=O)c(CNC)c[n]1[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', '5-methylaminomethyl-2-geranylthiouridine'),
    'l': Bases('l', 'U', 'AG', 'NC(Cc1c[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=S)[nH]c1=O)=O', '5-carbamoylmethyl-2-thiouridine'),
    'r': Bases('r', 'U', 'AG', 'NC(C(O)c1c[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]c1=O)=O', '5-carbamoylhydroxymethyluridine'),
    'y': Bases('y', 'G', 'CU', 'C[n]1c2nc(C)c(CC(O)C(N)C(OC)=O)[n]2c(=O)c2nc[n]([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)c12', 'methylated undermodified hydroxywybutosine'),
    '{': Bases('{', 'U', 'AG', 'CNCc1c(=O)[nH]c(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c1', '5-methylaminomethyluridine'),
    '|': Bases('|', 'G', 'CU', 'CN(C)c1[nH]c(=O)c2nc[n]([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3OC)c2n1', "N2,N2,2'-O-trimethylguanosine"),
    '}': Bases('}', 'C', 'G', 'N[C@H](C(=O)O)CCCCN=c1[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)ccc(N)n1', '2-lysidine'),
    '~': Bases('~', 'U', 'AG', 'CO[C@H]1[C@H]([n]2cc(CC(N)=O)c(=O)[nH]c2=O)O[C@H](CO)[C@H]1O', "5-carbamoylmethyl-2'-O-methyluridine"),
    '¡': Bases('¡', 'C', 'G', 'CO[C@H]1[C@H]([n]2cc(CO)c(N)nc2=O)O[C@H](CO)[C@H]1O', "2'-O-methyl-5-hydroxymethylcytidine"),
    '£': Bases('£', 'A', 'U', 'CC(C)=CCNc1nc(SCSC)nc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', '2- methylthiomethylenethio-N6-isopentenyl-adenosine'),
    '¥': Bases('¥', 'G', 'CU', 'Cc1nc2[nH]c3c(c(=O)[n]2c1CCC(N)C(O)=O)nc[n]3[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', '7-aminocarboxypropyl-demethylwyosine'),
    '«': Bases('«', 'A', 'U', 'OC(C(NC(Nc1ncnc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O)=O)C(O)CO)=O', 'hydroxy-N6-threonylcarbamoyladenosine'),
    '°': Bases('°', 'C', 'G', 'CO[C@H]1[C@H]([n]2cc(C=O)c(N)nc2=O)O[C@H](CO)[C@H]1O', "5-formyl-2'-O-methylcytidine"),
    '±': Bases('±', 'A', 'U', 'Cc1nc2c(N)nc(C)nc2[n]1[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', '2,8-dimethyladenosine'),
    'Ю': Bases('Ю', 'U', 'AG', 'CO[C@H]1[C@H]([n]2c(=O)[nH]c(=O)c(CNCC=C(C)C)c2)O[C@H](CO)[C@H]1O', "5-(isopentenylaminomethyl)-2'-O-methyluridine"),
    'Ɲ': Bases('Ɲ', 'U', 'AG', 'CC(=CCNCc1c(=O)[nH]c(=S)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c1)C', '5-(isopentenylaminomethyl)-2-thiouridine'),
    '¾': Bases('¾', 'U', 'AG', 'CC(=CCNCc1c(=O)[nH]c(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c1)C', '5-(isopentenylaminomethyl)uridine'),
    '¿': Bases('¿', 'C', 'G', 'NC(NCCCCNc1nc(=N)cc[n]1[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O)=N', 'agmatidine'),
    'Ç': Bases('Ç', 'C', 'G', 'Nc1nc(=O)[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)cc1O', '5-hydroxycytidine'),
    'ʭ': Bases('ʭ', 'U', 'AG', 'OC[C@H]1O[C@@H]([n]2cc(CNCCS(O)(=O)=O)c(=O)[nH]c2=O)[C@H](O)[C@@H]1O', '5-taurinomethyluridine'),
    'Ð': Bases('Ð', 'U', 'AG', 'NC(CCN1C(=O)N([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)CCC1=O)C(O)=O', '3-(3-amino-3-carboxypropyl)-5,6-dihydrouridine'),
    'Þ': Bases('Þ', 'U', 'AG', 'NC(C(=O)O)CC[n]1c(=O)c([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c[nH]c1=O', '3-(3-amino-3-carboxypropyl)pseudouridine'),
    'â': Bases('â', 'A', 'U', 'Cc1nc2c(N)ncnc2[n]1[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', '8-methyladenosine'),
    'æ': Bases('æ', 'G', 'CU', '[C@H]1([n]2c[n+](C)c3c2nc(nc3=O)NC)[C@H](OC)[C@H](O)[C@@H](CO)O1', "N2,7,2'-O-trimethylguanosine"),
    'ÿ': Bases('ÿ', 'A', 'U', 'CSc1nc2c(nc[n]2[C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c(C2C(=O)[C@H]([C@H](O)C)NC2=O)n1', '2-methylthio cyclic N6-threonylcarbamoyladenosine'),
    'œ': Bases('œ', 'A', 'U', 'CO[C@H]1[C@H]([n]2c3nc[n](c(=N)c3nc2)C)O[C@H](CO)[C@H]1O', "1,2'-O-dimethyladenosine"),
    'š': Bases('š', 'G', 'CU', 'C[n]1c2c(nc[n]2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[n]2c1nc(C)c2CC(O)C(N)C(O)=O', 'undermodified hydroxywybutosine'),
    'Γ': Bases('Γ', 'U', 'AG', 'CC(C)=CCC/C(/C)=C/CSc1nc(=O)cc[n]1[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', '2-geranylthiouridine'),
    'Δ': Bases('Δ', 'U', 'AG', 'CC(C)=CCC/C(/C)=C/CSc1nc(=O)c(CN)c[n]1[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', '5-aminomethyl-2-geranylthiouridine'),
    'Z': Bases('Z', 'U', 'AG', 'CO[C@H]1[C@H](c2c[nH]c(=O)[nH]c2=O)O[C@H](CO)[C@H]1O', "2'-O-methylpseudouridine"),
    'Ƒ': Bases('Ƒ', 'U', 'AG', 'C[n]1c(=O)c([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c[nH]c1=O', '3-methylpseudouridine'),
    'λ': Bases('λ', 'C', 'G', 'CO[C@H]1[C@H]([n]2ccc(NC)nc2=O)O[C@H](CO)[C@H]1O', "N4,2'-O-dimethylcytidine"),
    'Ω': Bases('Ω', 'G', 'CU', 'C[n]1c2c(nc[n]2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[n]2c1nc(C)c2CCC(N)C(O)=O', '7-aminocarboxypropylwyosine'),
    'α': Bases('α', 'U', 'AG', 'C[n]1c(=O)[n](CCC(C(=O)O)N)c(=O)c([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c1', '1-methyl-3-(3-amino-3-carboxypropyl)pseudouridine'),
    'β': Bases('β', 'C', 'G', 'CO[C@H]1[C@H]([n]2c(=O)nc(N(C)C)cc2)O[C@H](CO)[C@H]1O', "N4,N4,2'-O-trimethylcytidine"),
    'γ': Bases('γ', 'G', 'CU', 'CO[C@H]1[C@H]([n]2cnc3c2nc([nH]c3=O)NC)O[C@H](CO)[C@H]1O', "N2,2'-O-dimethylguanosine"),
    'δ': Bases('δ', 'U', 'AG', 'C[n]1c(=O)[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)ccc1=O', '3-methyluridine'),
    'ε': Bases('ε', 'G', 'CU', 'CO[C@H]1[C@H]([n]2c3nc([n](c(=O)c3nc2)C)N)O[C@H](CO)[C@H]1O', "1,2'-O-dimethylguanosine"),
    'ζ': Bases('ζ', 'A', 'U', 'CN(C)c1ncnc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', 'N6,N6-dimethyladenosine'),
    'η': Bases('η', 'A', 'U', 'CN(C)c1ncnc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1OC', "N6,N6,2'-O-trimethyladenosine"),
    'μ': Bases('μ', 'C', 'G', 'CN(c1nc(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)cc1)C', 'N4,N4-dimethylcytidine'),
    'ν': Bases('ν', 'C', 'G', 'CNc1nc(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)cc1', 'N4-methylcytidine'),
    'ξ': Bases('ξ', 'A', 'U', 'CO[C@H]1[C@H]([n]2c3nc[n](c(=O)c3nc2)C)O[C@H](CO)[C@H]1O', "1,2'-O-dimethylinosine"),
    'π': Bases('π', 'U', 'AG', 'NCc1c[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=[Se])[nH]c1=O', '5-aminomethyl-2-selenouridine'),
    'ρ': Bases('ρ', 'U', 'AG', 'CC1C(=O)NC(=O)N([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)C1', '5-methyldihydrouridine'),
    'ς': Bases('ς', 'G', 'CU', 'Nc1[nH]c(=O)c2c([n]([C@H]3[C@H](O)[C@H](O)[C@@H](CO)O3)cc2CN[C@@H]2[C@@H](O)[C@@H](O)C3C2O3)n1', 'epoxyqueuosine'),
    'σ': Bases('σ', 'U', 'AG', 'C[n]1c(=O)cc[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2OC)c1=O', "3,2'-O-dimethyluridine"),
    'τ': Bases('τ', 'C', 'G', 'CO[C@H]1[C@H]([n]2c(=O)nc(N)c(C)c2)O[C@H](CO)[C@H]1O', "5,2'-O-dimethylcytidine"),
    'υ': Bases('υ', 'U', 'AG', 'COC(COc1c[n](C2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]c1=O)=O', 'uridine 5-oxyacetic acid methyl ester'),
    'φ': Bases('φ', 'G', 'CU', 'Nc1[nH]c(=O)c2c(c[n]([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)c2n1)C#N', '7-cyano-7-deazaguanosine'),
    'χ': Bases('χ', 'A', 'U', 'CO[C@H]1[C@H]([n]2cnc3c2ncnc3NC)O[C@H](CO)[C@H]1O', "N6,2'-O-dimethyladenosine"),
    'ω': Bases('ω', 'U', 'AG', 'O[C@H]1[C@H]([n]2ccc(=O)[nH]c2=[Se])O[C@H](CO)[C@H]1O', '2-selenouridine'),
    'Ϩ': Bases('Ϩ', 'A', 'U', 'OC[C@H]1O[C@@H]([n]2cnc3c(NC=O)ncnc23)[C@H](O)[C@@H]1O', 'N6-formyladenosine'),
    'Ϫ': Bases('Ϫ', 'A', 'U', 'O[C@H]1[C@H]([n]2cnc3c(NCO)ncnc23)O[C@H](CO)[C@H]1O', 'N6-hydroxymethyladenosine'),
    'Ͽ': Bases('Ͽ', 'U', 'AG', 'COC(COc1c[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2OC)c(=O)[nH]c1=O)=O', "2'-O-methyluridine 5-oxyacetic acid methyl ester"),
    'Ѷ': Bases('Ѷ', 'U', 'AG', 'N#CCc1c[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]c1=O', '5-cyanomethyluridine'),
    '†': Bases('†', 'G', 'CU', 'Cc1c[n]2c(c3nc[n]([C@@H]4O[C@H](CO)[C@@H](O)[C@H]4O)c3[nH]c2n1)=O', '4-demethylwyosine'),
    '€': Bases('€', 'G', 'CU', 'C[n]1c2c(nc[n]2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[n]2cc(C)nc12', 'wyosine'),
    '℘': Bases('℘', 'U', 'AG', 'OC(Cc1c[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=S)[nH]c1=O)=O', '5-carboxymethyl-2-thiouridine'),
    'ℵ': Bases('ℵ', 'C', 'G', 'CO[C@H]1[C@H]([n]2c(=O)nc(NC(=O)C)cc2)O[C@H](CO)[C@H]1O', "N4-acetyl-2'-O-methylcytidine"),
    '⇑': Bases('⇑', 'G', 'CU', 'C[n]1c2nc(C)c(CCC(N)C(OC)=O)[n]2c(=O)c2nc[n]([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)c12', '7-aminocarboxypropylwyosine methyl ester'),
    '⇓': Bases('⇓', 'A', 'U', 'CC(Nc1ncnc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O)=O', 'N6-acetyladenosine'),
    'ƕ': Bases('ƕ', 'U', 'AG', 'OC[C@H]1O[C@@H]([n]2cc(CNCCS(O)(=O)=O)c(=O)[nH]c2=S)[C@H](O)[C@@H]1O', '5-taurinomethyl-2-thiouridine'),
    'Ƣ': Bases('Ƣ', 'C', 'G', 'Nc1nc(=O)[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)cc1CO', '5-hydroxymethylcytidine'),
    '∉': Bases('∉', 'G', 'CU', 'Nc1[nH]c(=O)c2c(CN)c[n]([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)c2n1', '7-aminomethyl-7-deazaguanosine'),
    '∏': Bases('∏', 'U', 'AG', 'CO[C@H]1[C@H]([n]2ccc(=O)[nH]c2=S)O[C@H](CO)[C@H]1O', "2-thio-2'-O-methyluridine"),
    '∑': Bases('∑', 'G', 'CU', 'Cc1c(C)[n]2c([n](c3c(c2=O)nc[n]3[C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)C)n1', 'methylwyosine'),
    '√': Bases('√', 'A', 'U', 'CCC(O)C(NC(Nc1ncnc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O)=O)C(O)=O', 'N6-hydroxynorvalylcarbamoyladenosine'),
    '∝': Bases('∝', 'U', 'AG', 'OC[C@@H]1[C@@H](O)[C@@H](O)[C@H]([n]2c(=O)[nH]c(=O)c(O)c2)O1', '5-hydroxyuridine'),
    '∞': Bases('∞', 'A', 'U', 'CNc1nc(SC)nc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', '2-methylthio-N6-methyladenosine'),
    '∠': Bases('∠', 'G', 'CU', 'CN(c1nc(=O)c2c([n]([C@H]3[C@H](O)[C@H](O)[C@@H](CO)O3)c[n+]2C)n1)C', 'N2,N2,7-trimethylguanosine'),
    '∨': Bases('∨', 'G', 'CU', 'CNc1nc(=O)c2c([n]([C@H]3[C@H](O)[C@H](O)[C@@H](CO)O3)c[n+]2C)n1', 'N2,7-dimethylguanosine'),
    '∩': Bases('∩', 'U', 'AG', 'CO[C@H]1[C@H]([n]2c(=O)[nH]c(=O)c(CC(OC)=O)c2)O[C@H](CO)[C@H]1O', "5-methoxycarbonylmethyl-2'-O-methyluridine"),
    '∪': Bases('∪', 'U', 'AG', 'NCc1c(=O)[nH]c(=O)[n]([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)c1', '5-aminomethyluridine'),
    '∫': Bases('∫', 'U', 'AG', 'NCc1c[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=S)[nH]c1=O', '5-aminomethyl-2-thiouridine'),
    '≅': Bases('≅', 'U', 'AG', 'CNCc1c[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=[Se])[nH]c1=O', '5-methylaminomethyl-2-selenouridine'),
    '≈': Bases('≈', 'A', 'U', 'CCC(O)C(NC(Nc1nc(SC)nc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O)=O)C(O)=O', '2-methylthio-N6-hydroxynorvalylcarbamoyladenosine'),
    '≠': Bases('≠', 'A', 'U', 'CC(CO)=CCNc1nc(SC)nc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', '2-methylthio-N6-(cis-hydroxyisopentenyl) adenosine'),
    '≡': Bases('≡', 'A', 'U', 'OC(CNC(Nc1ncnc2c1nc[n]2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O)=O)=O', 'N6-glycinylcarbamoyladenosine'),
    'Ш': Bases('Ш', 'A', 'U', 'CO[C@H]1[C@H]([n]2cnc3c2nc[nH]c3=O)O[C@H](CO)[C@H]1O', "2'-O-methylinosine"),
    '≥': Bases('≥', 'U', 'AG', 'OC[C@@H]1[C@@H](O)[C@@H](O)[C@H]([n]2c(=O)[nH]c(=O)c(C(C(=O)O)O)c2)O1', '5-carboxyhydroxymethyluridine'),
    '⊄': Bases('⊄', 'G', 'CU', 'NC(C(O[C@@H]1[C@@H](NCc2c3c(nc(nc3=O)N)[n]([C@H]3[C@H](O)[C@H](O)[C@@H](CO)O3)c2)C=C[C@@H]1O)=O)CCC(=O)O', 'glutamyl-queuosine'),
    '⊆': Bases('⊆', 'G', 'CU', 'C[n]1c2c(nc[n]2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[n]2c1nc(C)c2CC(O)C(NC(OC)=O)C(OC)=O', 'hydroxywybutosine'),
    '⊇': Bases('⊇', 'G', 'CU', 'Cc1nc2[nH]c3c(c(=O)[n]2c1C)nc[n]3[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O', 'isowyosine'),
    '⊥': Bases('⊥', 'U', 'AG', 'OC(CNCc1c[n]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=[Se])[nH]c1=O)=O', '5-carboxymethylaminomethyl-2-selenouridine'),
    '◊': Bases('◊', 'U', 'AG', 'OC[C@@H]1[C@@H](O)[C@@H](O)[C@H]([n]2c(=O)[nH]c(=O)c(CC(=O)O)c2)O1', '5-carboxymethyluridine'),
}

# The currently active nucleoside dictionary.
# Default: modomics_nucleosides (full MODOMICS catalog)
# When --modchar-vienna-compat is used: supported_nucleosides
active_nucleosides: Dict[str, Bases] = modomics_nucleosides


def set_active_nucleosides(vienna_compat: bool = False) -> None:
    """Switch the active nucleoside dictionary.

    Args:
        vienna_compat: If True, use supported_nucleosides (Vienna-compatible subset).
                       If False (default), use modomics_nucleosides.
    """
    global active_nucleosides
    active_nucleosides = supported_nucleosides if vienna_compat else modomics_nucleosides
