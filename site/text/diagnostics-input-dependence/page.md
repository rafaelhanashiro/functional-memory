> [!note] Calibration-mean intervention
> - Compute each head's average coefficients over a calibration set (64 validation sequences).
> - Fix every head's polynomial, in all layers at once, to these coefficients.
> - Evaluate the model on 256 other validation sequences.
