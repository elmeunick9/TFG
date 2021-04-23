# Extension Proposal List

## Main Extensions

Focus on improving the performance of the selection.

- **Non-linear Kernel**: Use Polynomial, Gaussian, RBF, or other such kernels. New ranking criteria needs to be found for the kernel selected. [1]
- **Avoid using cross-validation**: To be defined. *Remainder: using C parameter). See also [2], is this related?*
- **Using historic of weights**: Keep information on the ranking of each feature (weights) for previous iterations and use it in combination with the current iteration weight vector to make a better ranking criteria.
- **Multi-class criteria**: How do we handle an array of weight vectors?
  - By selecting the feature to be discarded for each vector, and then taking the one with most rank (repeated most times). *Problem: Can lead to cases where all selected features are different, but: by using step > 1 and multiply by position this can be mitigated, also most features should be the same since they are all (supposedly) non-relevant.*
  - By selecting minimum of them all, or better, select the minimum of them all but normalize the vectors first using the amount of information in each class-to-class comparison (amount of observations).

## Secondary Extensions

Focus on reducing the computational cost.

 - **Sampling**: Use a subset of the observations. Ideally, use a different subset on each iteration, see if results converge. MSVM-RFE (multiple SVM with sampling in each iteration [3]) could be investigated, but it is not a priority.
 - **Dynamic Step**: Instead of using a constant value in each iteration, calculate it dynamically by:
   - Use a percentage on the amount of remaining features.
   - Use the difference in scores from the last iteration.
 - **Stop condition**: Determine the amount of features that are relevant (SVM-RFE only provides a ranking). *Cross-Validation is used here. Isn't this related to the "Avoid using cross-validation" extension?*

# Kanban

| Extension Name | TO DO | DEFINITION | IMPLEMENTATION | COMPLETED
| :- | -: | -: | -: | -: |
| Non-linear Kernel | X | | | |
| Avoid CV | X | | | |
| Historic of weights | X | | | |
| Multi-class criteria | X | | | |
| Sampling | X | | | |
| Dynamic Step | |  | | X |
| Stop condition | | X | | |

# Bibliography

[1] Nonlinear feature selection using Gaussian kernel SVM-RFE for fault diagnosis

[2] SVM-RFE Based Feature Selection and Taguchi Parameters Optimization for Multiclass SVM Classifier

[3] Classification of lip color based on multiple SVM-RFE