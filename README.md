# Baseball ML: Team Wins & Hall of Fame

PyTorch regression and classification models trained on the Lahman Baseball Database, exported to ONNX and run in the browser with onnxruntime-web.

**Live demo:** https://davidgarcia355.github.io/baseball-ml/

| Task | Notebook | Model file | Result |
|---|---|---|---|
| Regression: predict MLB team wins | `Regression_TeamWins_PyTorch.ipynb` | `wins_regression.onnx` | Test R² = 0.900 |
| Classification: predict Hall of Fame induction | `Classification_HallOfFame_PyTorch.ipynb` | `hof_classifier.onnx` | Test F1 = 0.793 |

Both ONNX models take `input1` with shape [batch, 8] (raw stats) and return `output1`.
