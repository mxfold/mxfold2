import unittest

import torch
import torch.nn as nn

from mxfold2.loss.shape_fy_loss import ShapeFenchelYoungLoss


class RecordingFold(nn.Module):
    """Minimal fold stub that distinguishes selection from rescoring."""

    def __init__(self):
        super().__init__()
        self.score = nn.Parameter(torch.tensor(1.0))
        self.score_weight_turner = 0.25
        self.score_weight_positional = 0.75
        self.seen_pseudoenergy = []
        self.selection_weights = []
        self.constraint_calls = 0

    def forward(
        self,
        seq,
        return_param=False,
        perturb=0.0,
        param=None,
        pseudoenergy=None,
        constraint=None,
    ):
        structures = [torch.zeros(len(s) + 1, dtype=torch.long) for s in seq]

        if return_param:
            scores = torch.stack([2.0 * self.score for _ in seq])
            return scores, ["." * len(s) for s in seq], structures, [{}], [{}]

        if pseudoenergy is not None:
            self.seen_pseudoenergy.append([value.detach().clone() for value in pseudoenergy])
            self.selection_weights.append(
                (self.score_weight_turner, self.score_weight_positional)
            )
            scores = torch.stack(
                [self.score + value.sum() for value in pseudoenergy]
            )
            return scores, ["." * len(s) for s in seq], structures

        if constraint is not None:
            self.constraint_calls += 1
            scores = torch.stack([self.score for _ in seq])
            return scores, ["." * len(s) for s in seq], constraint

        raise AssertionError("unexpected fold call")


class ShapeFenchelYoungLossTest(unittest.TestCase):
    def test_forward_aligns_shape_values_and_does_not_mutate_target(self):
        model = RecordingFold()
        loss_fn = ShapeFenchelYoungLoss(model, perturb=0.0)
        target = torch.tensor([-999.0, 0.0, 0.5, 1.0])
        original = target.clone()

        first_loss = loss_fn(["ACG"], [target])
        second_loss = loss_fn(["ACG"], [target])

        torch.testing.assert_close(target, original)
        self.assertEqual(model.seen_pseudoenergy[0][0].numel(), 3)
        expected = torch.tensor([
            -0.8,
            2.6 * torch.log(torch.tensor(1.5)).item() - 0.8,
            2.6 * torch.log(torch.tensor(2.0)).item() - 0.8,
        ])
        torch.testing.assert_close(model.seen_pseudoenergy[0][0], expected)
        torch.testing.assert_close(
            model.seen_pseudoenergy[1][0], model.seen_pseudoenergy[0][0]
        )
        torch.testing.assert_close(first_loss, second_loss)

    def test_reference_structure_is_rescored_without_pseudoenergy(self):
        model = RecordingFold()
        loss_fn = ShapeFenchelYoungLoss(model, perturb=0.0)

        loss = loss_fn(
            ["ACG"],
            [torch.tensor([-999.0, 0.0, 0.5, 1.0])],
        )

        # Prediction score is 2, while the selected reference structure has
        # base score 1. The pseudoenergy-augmented selection score must not be
        # used as the reference score in the FY value.
        torch.testing.assert_close(loss, torch.tensor([1.0 / 3.0]))
        self.assertEqual(model.constraint_calls, 1)

    def test_invalid_shape_length_is_rejected(self):
        loss_fn = ShapeFenchelYoungLoss(RecordingFold(), perturb=0.0)

        with self.assertRaisesRegex(ValueError, "SHAPE target length"):
            loss_fn(["ACG"], [torch.tensor([0.0, 1.0])])

    def test_scheduled_weights_are_used_only_for_reference_selection(self):
        model = RecordingFold()
        loss_fn = ShapeFenchelYoungLoss(
            model,
            perturb=0.0,
            weight_schedule="linear",
            weight_schedule_start=1,
            weight_schedule_end=2,
        )
        loss_fn.set_epoch_info(epoch=2, total_epochs=2)

        loss = loss_fn(
            ["ACG"],
            [torch.tensor([-999.0, 0.0, 0.5, 1.0])],
        )

        self.assertEqual(model.selection_weights, [(0.5, 0.5)])
        self.assertEqual(model.score_weight_turner, 0.25)
        self.assertEqual(model.score_weight_positional, 0.75)
        torch.testing.assert_close(loss, torch.tensor([1.0 / 3.0]))


if __name__ == "__main__":
    unittest.main()
