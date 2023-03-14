import dataclasses

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