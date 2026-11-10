# Full bounded pointer local evaluation

Status: **complete**.

1000 test questions; settings inherited unchanged from pilot development. No new development stage.
Some source documents overlap prior evaluations; question freshness does not establish unseen-document generalization.
Whole-input budgets: 1,024, 2,048 and 4,096. NF4, batch one; native non-thinking chat.
Pointer caps are chosen on development data only. All stage predictions, per-task metrics and paired confidence intervals are retained.
No overall score pools different metrics; report coverage and per-task paired uncertainty.

| Stage | Recorded statuses |
| --- | --- |
| test_1024 | {'scored': 2922, 'skipped': 1078} |
| test_2048 | {'scored': 3319, 'skipped': 681} |
| test_4096 | {'scored': 3700, 'skipped': 300} |

## Paired test comparisons

{
  "test_1024": {
    "adaptive_no_index": {
      "n_paired": 972,
      "wins": 111,
      "ties": 727,
      "losses": 134,
      "mean_input_token_difference": 1.0884773662551441,
      "generation_ms_difference_total": -52910.28355184244
    },
    "bm25_budget": {
      "n_paired": 972,
      "wins": 111,
      "ties": 727,
      "losses": 134,
      "mean_input_token_difference": -0.9002057613168725,
      "generation_ms_difference_total": -53961.03070769459
    },
    "original_sempointer": {
      "n_paired": 6,
      "wins": 1,
      "ties": 5,
      "losses": 0,
      "mean_input_token_difference": -30.333333333333332,
      "generation_ms_difference_total": -315.59059012215585
    }
  },
  "test_2048": {
    "adaptive_no_index": {
      "n_paired": 1000,
      "wins": 130,
      "ties": 739,
      "losses": 131,
      "mean_input_token_difference": 51.145,
      "generation_ms_difference_total": -28497.044156538323
    },
    "bm25_budget": {
      "n_paired": 1000,
      "wins": 132,
      "ties": 686,
      "losses": 182,
      "mean_input_token_difference": -365.124,
      "generation_ms_difference_total": -179278.1906579039
    },
    "original_sempointer": {
      "n_paired": 319,
      "wins": 27,
      "ties": 245,
      "losses": 47,
      "mean_input_token_difference": -323.4576802507837,
      "generation_ms_difference_total": -43186.107963498216
    }
  },
  "test_4096": {
    "adaptive_no_index": {
      "n_paired": 1000,
      "wins": 134,
      "ties": 740,
      "losses": 126,
      "mean_input_token_difference": 109.135,
      "generation_ms_difference_total": -19746.726545970887
    },
    "bm25_budget": {
      "n_paired": 1000,
      "wins": 123,
      "ties": 648,
      "losses": 229,
      "mean_input_token_difference": -2172.445,
      "generation_ms_difference_total": -895749.3470078916
    },
    "original_sempointer": {
      "n_paired": 700,
      "wins": 83,
      "ties": 523,
      "losses": 94,
      "mean_input_token_difference": -515.2971428571428,
      "generation_ms_difference_total": -132521.79484604858
    }
  }
}
