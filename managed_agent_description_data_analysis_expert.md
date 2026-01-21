I am a managed agent with name data_analysis_expert. I can assist with code-related tasks.

## Available npz files
You have access to the following NPZ data files:
 <npz_0> - `data_all/ATVA/ball/sample_0.npz`
 <npz_1> - `data_all/ATVA/ball/sample_1.npz`
 <npz_2> - `data_all/ATVA/ball/sample_2.npz`

**Quick Access via State Variables**
- `DATA_FILE_PATHS`: List of all available NPZ file paths
- `DATA_FILE_PATH`: Path to the first/primary data file (for convenience)

**Example Usage**
```python
import numpy as np
# Load a specific file
data = np.load(DATA_FILE_PATHS[0])
# Or use the primary file
data = np.load(DATA_FILE_PATH)
```

