"""Modular Structure Migration Guide

This document outlines the modularization of xbpm_bumps.py

## Directory Structure

```
XBPM-bumps/
├── xbpm_bumps/                 # Main package
│   ├── __init__.py             # Package exports
│   ├── core/                   # Business logic
│   │   ├── __init__.py
│   │   ├── config.py           # Config class (DONE)
│   │   ├── constants.py        # Module constants (DONE)
│   │   ├── parameters.py       # Prm, ParameterBuilder
│   │   ├── readers.py          # DataReader
│   │   ├── processors.py       # XBPMProcessor, BPMProcessor
│   │   ├── visualizers.py      # BladeMapVisualizer, PositionVisualizer
│   │   └── exporters.py        # Exporter
│   └── ui/                     # PyDM interface (future)
│       ├── __init__.py
│       ├── main_window.py
│       └── widgets/
├── xbpm_bumps.py               # CLI entry point (backward compat)
└── README.md

```

## Migration Steps

1. ✅ Create directory structure
2. ✅ Create Config class in config.py
3. ✅ Create constants.py
4. ⏳ Extract parameters.py (Prm + ParameterBuilder)
5. ⏳ Extract readers.py (DataReader)
6. ⏳ Extract processors.py (XBPMProcessor + BPMProcessor)
7. ⏳ Extract visualizers.py (BladeMapVisualizer + PositionVisualizer)
8. ⏳ Extract exporters.py (Exporter)
9. ⏳ Create new CLI entry point that imports from package
10. ⏳ Create XBPMApp orchestrator in package

## Next Steps for UI

After modularization is complete:
- Create ui/ directory
- Design PyDM main window layout
- Create custom widgets
- Implement Qt signal/slot architecture
- Add async processing for heavy calculations
"""
