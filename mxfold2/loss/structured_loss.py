from mxfold2.loss.fy_loss import FenchelYoungLoss

# Backward compatibility alias.
# StructuredLoss is now unified into FenchelYoungLoss.
# When loss_pos_paired/loss_neg_paired are non-zero, it behaves as structured hinge loss.
StructuredLoss = FenchelYoungLoss
