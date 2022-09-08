import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalScore(nn.Module):
    def __init__(self, embedding: torch.Tensor, bilinears: nn.ModuleDict, fc_length: dict[str, torch.Tensor]):
        super(PositionalScore, self).__init__()
        self.embedding = embedding # (N, C)
        self.bl_helix_stacking = bilinears['helix_stacking']
        self.bl_mismatch_hairpin = bilinears['mismatch_hairpin']
        self.bl_mismatch_multi = bilinears['mismatch_multi']
        self.bl_mismatch_internal = bilinears['mismatch_internal']
        self.bl_mismatch_external = bilinears['mismatch_external']
        self.bl_base_hairpin = bilinears['base_hairpin']
        self.bl_base_multi = bilinears['base_multi']
        self.bl_base_internal = bilinears['base_internal']
        self.bl_base_external = bilinears['base_external']
        self.score_hairpin_length = fc_length['score_hairpin_length']
        self.score_bulge_length = fc_length['score_bulge_length']
        self.score_internal_length = fc_length['score_internal_length']
        self.score_internal_explicit = fc_length['score_internal_explicit']
        self.score_internal_symmetry = fc_length['score_internal_symmetry']
        self.score_internal_asymmetry = fc_length['score_internal_asymmetry']
        self.score_helix_length = fc_length['score_helix_length']
        self.total_energy = 0

    def score_hairpin(self, i: int, j: int) -> torch.Tensor:
        l = (j-1)-(i+1)+1
        e = 0.
        e += self.score_hairpin_length[min(l, len(self.score_hairpin_length)-1)].reshape(1,)
        e += self.score_base_hairpin_(i+1, j-1)
        e += self.score_mismatch_hairpin_(i, j)
        e += self.score_basepair_(i, j)
        return e

    def count_hairpin(self, i: int, j: int) -> None:
        self.total_energy += self.score_hairpin(i, j)

    
    def score_base_hairpin_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_base_hairpin(self.embedding[i], self.embedding[j])

    def score_mismatch_hairpin_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_mismatch_hairpin(self.embedding[i], self.embedding[j])

    def score_basepair_(self, i: int, j: int) -> torch.Tensor:
        return 0


    def score_single_loop(self, i: int, j: int, k: int, l: int) -> torch.Tensor:
        l1 = (k-1)-(i+1)+1
        l2 = (j-1)-(l+1)+1
        ls, ll = min(l1, l2), max(l1, l2)
        e = 0.

        if ll==0: # stack
            e += self.score_helix_stacking_(i, j)
            e += self.score_helix_stacking_(l, k)
            e += self.score_basepair_(i, j)
            return e

        elif ls==0: # bulge
            e += self.score_bulge_length[min(ll, len(self.score_bulge_length)-1)].reshape(1,)
            e += self.score_base_internal_(i+1, k-1) + self.score_base_internal_(l+1, j-1)
            e += self.score_mismatch_internal_(i, j) + self.score_mismatch_internal_(l, k)
            e += self.score_basepair_(i, j)
            return e

        else: # internal loop
            e += self.score_internal_length[min(ls+ll, len(self.score_internal_length)-1)].reshape(1,)
            e += self.score_base_internal_(i+1, k-1) 
            e += self.score_base_internal_(l+1, j-1)
            e += self.score_internal_explicit[min(ls, len(self.score_internal_explicit)-1), min(ll, len(self.score_internal_explicit)-1)].reshape(1,)
            if ls==ll:
                e += self.score_internal_symmetry[min(ll, len(self.score_internal_symmetry)-1)].reshape(1,)
            e += self.score_internal_asymmetry[min(ll-ls, len(self.score_internal_asymmetry)-1)].reshape(1,)
            e += self.score_mismatch_internal_(i, j) + self.score_mismatch_internal_(l, k)
            e += self.score_basepair_(i, j)
            return e

    def count_single_loop(self, i: int, j: int, k: int, l: int) -> None:
        self.total_energy += self.score_single_loop(i, j, k, l)

    def score_helix_stacking_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_helix_stacking(self.embedding[i], self.embedding[j])

    def score_base_internal_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_base_internal(self.embedding[i], self.embedding[j])

    def score_mismatch_internal_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_mismatch_internal(self.embedding[i], self.embedding[j])


    def score_helix(self, i: int, j: int, m: int) -> torch.Tensor:
        e = 0.
        e += self.score_helix_length[min(m, len(self.score_helix_length)-1)].reshape(1,)
        for k in range(1, m):
            e += self.score_helix_stacking_(i+(k-1), j-(k-1))
            e += self.score_helix_stacking_(j-k, i+k)
            e += self.score_basepair_(i+(k-1), j-(k-1));
        return e

    def count_helix(self, i: int, j: int, m: int) -> None:
        self.total_energy += self.score_helix(i, j, m)


    def score_multi_loop(self, i: int, j: int) -> torch.Tensor:
        return self.score_mismatch_multi_(i, j) + self.score_basepair_(i, j)

    def count_multi_loop(self, i: int, j: int) -> None:
        self.total_energy += self.score_multi_loop(i, j)

    def score_mismatch_multi_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_mismatch_multi(self.embedding[i], self.embedding[j])
    

    def score_multi_paired(self, i: int, j: int) -> torch.Tensor:
        return self.score_mismatch_multi_(j, i)

    def count_multi_paired(self, i: int, j: int) -> None:
        self.total_energy += self.score_multi_paired(i, j)


    def score_multi_unpaired(self, i: int, j: int) -> torch.Tensor:
        return self.score_base_multi_(i, j)
    
    def count_multi_unpaired(self, i: int, j: int) -> None:
        self.total_energy += self.score_multi_unpaired(i, j)

    def score_base_multi_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_base_multi(self.embedding[i], self.embedding[j])


    def score_external_paired(self, i: int, j: int) -> torch.Tensor:
        return self.score_mismatch_external_(j, i)

    def count_external_paired(self, i: int, j: int) -> None:
        self.total_energy += self.score_external_paired(i, j)

    def score_mismatch_external_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_mismatch_external(self.embedding[i], self.embedding[j])
        

    def score_external_unpaired(self, i: int, j: int) -> torch.Tensor:
        return self.score_base_external_(i, j)

    def count_external_unpaired(self, i: int, j: int) -> None:
        self.total_energy += self.score_external_unpaired(i, j)

    def score_base_external_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_base_external(self.embedding[i], self.embedding[j])

