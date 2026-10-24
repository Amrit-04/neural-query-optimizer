# Error Analysis Report

- **Total Plans Analyzed**: 280

- **Mean Absolute Error**: 2207.45 ms

- **Median Absolute Error**: 1268.06 ms

- **Max Absolute Error**: 5999.59 ms

- **Mean Relative Error**: 99.74%


## Worst Predictions:

- **Plan plan_q01_p04.json** (Query: 1):
  - **Actual Cost**: 6000.00
  - **Predicted**: 0.41
  - **Absolute Error**: 5999.59
  - **Relative Error**: 99.99%
  - **Plan Depth**: 4
  - **Operators**: Sort, Merge Join, Hash Join, Nested Loop, Index Scan, Index Scan, Hash Join, Index Scan, Index Scan, Index Scan

- **Plan plan_q31_p05.json** (Query: 31):
  - **Actual Cost**: 6000.00
  - **Predicted**: 0.46
  - **Absolute Error**: 5999.54
  - **Relative Error**: 99.99%
  - **Plan Depth**: 3
  - **Operators**: Hash Join, Index Scan, Hash Join, Index Scan, Hash Join, Index Scan, Index Scan

- **Plan plan_q10_p07.json** (Query: 10):
  - **Actual Cost**: 6000.00
  - **Predicted**: 0.46
  - **Absolute Error**: 5999.54
  - **Relative Error**: 99.99%
  - **Plan Depth**: 3
  - **Operators**: Nested Loop, Nested Loop, Index Scan, Index Scan, Nested Loop, Index Scan, Nested Loop, Index Scan, Index Scan

- **Plan plan_q08_p07.json** (Query: 8):
  - **Actual Cost**: 6000.00
  - **Predicted**: 0.47
  - **Absolute Error**: 5999.53
  - **Relative Error**: 99.99%
  - **Plan Depth**: 3
  - **Operators**: Nested Loop, Index Scan, Nested Loop, Index Scan, Nested Loop, Index Scan, Index Scan

- **Plan plan_q18_p00.json** (Query: 18):
  - **Actual Cost**: 6000.00
  - **Predicted**: 0.52
  - **Absolute Error**: 5999.48
  - **Relative Error**: 99.99%
  - **Plan Depth**: 5
  - **Operators**: Sort, Aggregate, Hash Join, Hash Join, Bitmap Heap Scan, Index Scan, Hash Join, Index Scan, Hash Join, Index Scan, Seq Scan