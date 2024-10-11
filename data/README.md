# Study data
> Please explain here shortly the content of the data folders

For reproducibility the data should be strictly separated in `raw`, `processed` and `interim`.
There should be clear path how to get the preprocessed data in `interim` based on the `raw` data and how to get the final `processed` results from the other two. In the optimal case, there is a single (python) script that produces the resulting data based on the raw data.

*Raw* data should never be changed, because otherwise your results are not reproducible. Science is based on reproducibiltity which is why we value it highly. 