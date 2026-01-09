# HDF5 Data Preservation Analysis: Pickle vs HDF5

## Problem Statement

When importing HDF5 files and attempting to re-run analysis, an `IndexError` occurs in `processors.py:_tangents_calc()`:

```
IndexError: tuple index out of range
if dt[2]['agx'] == 0 and dt[2]['agy'] == 0:
   ~~^^^
```

This indicates that `rawdata` entries don't have the expected 3-element tuple structure `(?, ?, BPM_data_dict)`.

## Pickle File Structure (Current Implementation)

### Raw Data Format
From pickle files, `rawdata` is a list of tuples with structure:
```python
[(header_meta, nominal_grid_data),
 (header_meta, nominal_grid_data),
 ...]
```

**Each tuple contains 3 elements:**

1. **Element 0: Header Metadata** (dict-like)
   - Beamline information
   - Acquisition timestamp
   - Other metadata

2. **Element 1: Nominal Grid/Blade Data** (dict)
   - Keys: `(x_nom, y_nom)` tuples
   - Values: `[[to_mean, to_err], [ti_mean, ti_err], [bi_mean, bi_err], [bo_mean, bo_err]]`
   - Blade measurements at each position

3. **Element 2: BPM Data** (dict) - **CRITICAL FOR ANALYSIS**
   - Keys: Various parameter names
   - **Accessed keys in analysis:**
     - `'agx'`, `'agy'` - Angular corrections (used in tangent calculations)
     - `'orbx'`, `'orby'` - Orbit data arrays (indexed by sector)
     - `'beamline'` - Beamline identifier
     - `'datetime'` or timestamp fields
   - **This is the data being lost in HDF5 conversion!**

## Current HDF5 Storage Structure

### What IS Stored
From `exporters.py:write_hdf5()`:

```
/meta/               - Metadata (version, timestamp, beamline)
/parameters/         - Run parameters (as attributes)
/raw_data/
  ├─ measurements    - Structured array (x_nom, y_nom, blade values)
/analysis/
  ├─ positions/      - Position calculation results
  ├─ bpm_stats/      - BPM statistics
  ├─ sweeps/         - Sweep analysis data
  └─ figures/        - Pickled figure objects
```

### What IS NOT Stored
**The entire Element 2 (BPM data dictionary) is missing!**

This contains crucial information:
- Angular corrections (`agx`, `agy`)
- Orbit data (`orbx`, `orby`)
- Machine parameters
- Potentially undocumented keys needed for reprocessing

## Critical Code Paths Affected

### 1. **BPM Position Calculation** (`processors.py:calculate_positions()`)
```python
def _tangents_calc(self, sector_idx):
    for dt in self.rawdata:  # dt = (metadata, grid_data, BPM_dict)
        if dt[2]['agx'] == 0 and dt[2]['agy'] == 0:  # FAILS - dt[2] is None
            # Use this as reference orbit
```

### 2. **Offset Calculation** (`_offset_search()`)
```python
agx = np.array([dt[2]['agx'] for dt in self.rawdata])  # FAILS
orbx = np.array([dt[2]['orbx'][idx] for dt in self.rawdata])  # FAILS
```

### 3. **Central Sweep Analysis**
Multiple accesses to `dt[2]['agx']`, `dt[2]['agy']`, etc.

## Data Dependency Chain

```
Pickle Files
    ↓
DataReader._get_pickle_data()
    ↓
rawdata = [(meta, grid, bpm_dict), ...]
    ↓
BPMProcessor.__init__(rawdata)
    ↓
Analysis Steps:
  - calculate_positions() → needs dt[2] BPM dict
  - central_sweep_analysis() → needs dt[2] BPM dict
  - blade_map() → needs dt[2] BPM dict
```

When loading from HDF5:
```
HDF5 File
    ↓
DataReader.load_figures_from_hdf5()
    ↓
rawdata = None or [(meta, grid, None), ...]  ← Missing dt[2]!
    ↓
BPMProcessor attempts analysis with incomplete data
    ↓
CRASHES
```

## Strategy for Comprehensive HDF5 Storage

### Option 1: Store Raw BPM Data as Structured Dataset (RECOMMENDED)

**Advantages:**
- Preserves all information
- Allows complete reprocessing
- Maintains scientific reproducibility

**Implementation:**

```python
# In exporters.py:write_hdf5()

# New group for raw BPM monitoring data
bpm_grp = h5.create_group('bpm_data')

for i, (metadata, grid_data, bpm_dict) in enumerate(rawdata):
    # Store each sweep's BPM data
    sweep_grp = bpm_grp.create_group(f'sweep_{i:04d}')
    
    # Store metadata
    for key, val in metadata.items():
        if val is not None:
            try:
                sweep_grp.attrs[f'meta_{key}'] = val
            except TypeError:
                sweep_grp.attrs[f'meta_{key}'] = str(val)
    
    # Store BPM parameters
    for key, val in bpm_dict.items():
        try:
            if isinstance(val, (list, np.ndarray)):
                sweep_grp.create_dataset(key, data=np.asarray(val))
            elif isinstance(val, (int, float, str)):
                sweep_grp.attrs[key] = val
            else:
                sweep_grp.attrs[key] = str(val)
        except Exception as e:
            logger.warning(f"Could not store bpm_dict['{key}']: {e}")
```

### Option 2: Store as JSON String (Simple, Less Efficient)

```python
# Store entire BPM dict as JSON
import json
bpm_json = json.dumps(str(bpm_dict))  # or use json serialization
bpm_grp.attrs['bpm_data_json'] = bpm_json
```

### Option 3: Hybrid - Identified Essential Fields

**If Option 1 is too verbose, identify which fields are actually needed:**

Create a mapping of essential BPM parameters:
```python
ESSENTIAL_BPM_FIELDS = [
    'agx', 'agy',           # Angular corrections
    'orbx', 'orby',         # Orbit data
    'beamline',             # Beamline ID
    'datetime', 'timestamp', # Timing
    # Add others as discovered
]
```

Store only essential fields to reduce file size.

## Reader Side Implementation

### Option 1: Reconstruct rawdata from HDF5

```python
# In readers.py:load_figures_from_hdf5()

def _reconstruct_rawdata_from_hdf5(self, h5_file):
    """Reconstruct rawdata tuple list from HDF5."""
    rawdata = []
    
    if 'bpm_data' not in h5_file:
        logger.warning("No BPM data in HDF5; analysis may be incomplete")
        return rawdata
    
    bpm_grp = h5_file['bpm_data']
    for key in sorted(bpm_grp.keys()):
        sweep_grp = bpm_grp[key]
        
        # Reconstruct metadata dict
        metadata = {}
        for attr_key, attr_val in sweep_grp.attrs.items():
            if attr_key.startswith('meta_'):
                metadata[attr_key[5:]] = attr_val
        
        # Reconstruct nominal grid data (can use /raw_data/measurements)
        grid_data = self._load_nominal_grid()
        
        # Reconstruct BPM dict
        bpm_dict = {}
        for dataset_key in sweep_grp.keys():
            bpm_dict[dataset_key] = np.array(sweep_grp[dataset_key])
        
        for attr_key in sweep_grp.attrs.keys():
            if not attr_key.startswith('meta_'):
                bpm_dict[attr_key] = sweep_grp.attrs[attr_key]
        
        rawdata.append((metadata, grid_data, bpm_dict))
    
    return rawdata
```

## Implementation Steps

1. **Phase 1: Add BPM Data Storage to Exporter**
   - Modify `exporters.py:write_hdf5()` to save BPM data
   - Choose Option 1 (structured) for maximum flexibility

2. **Phase 2: Add BPM Data Reading to Reader**
   - Modify `readers.py:load_figures_from_hdf5()` to reconstruct rawdata
   - Ensure backward compatibility with old HDF5 files

3. **Phase 3: Update Analysis Flow**
   - Ensure `BPMProcessor` receives properly reconstructed rawdata
   - Add validation to detect incomplete rawdata early

4. **Phase 4: Testing**
   - Test round-trip: pickle → HDF5 → analysis produces same results
   - Test backward compatibility with existing HDF5 files

## Backward Compatibility

For existing HDF5 files without BPM data:
- Detect missing BPM data group
- Provide fallback warning message
- Disable re-analysis but allow figure viewing

## Files to Modify

1. **`xbpm_bumps/core/exporters.py`** - `write_hdf5()` method
2. **`xbpm_bumps/core/readers.py`** - `load_figures_from_hdf5()` and new helper methods
3. **`xbpm_bumps/ui/analyzer.py`** - Error handling for missing rawdata
4. **`xbpm_bumps/ui/main_window.py`** - User messaging about reanalysis capability

## Summary

**Root Cause:** HDF5 export only stores processed results and blade measurements, not the raw BPM monitoring data (Element 2 of rawdata tuples) needed for reprocessing.

**Solution:** Store the complete rawdata structure in HDF5, enabling full re-analysis without original pickle files.

**Recommended Approach:** Option 1 (structured HDF5 groups) for maximum flexibility and future extensibility.
