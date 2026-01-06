#!/usr/bin/env python3
"""Script to modularize xbpm_bumps.py into separate modules.

This script extracts classes and functions from the monolithic xbpm_bumps.py
and reorganizes them into a proper package structure.
"""

import re
from pathlib import Path


def extract_class(content: str, class_name: str) -> str:
    """Extract a class definition and its methods from content."""
    pattern = rf'^class {class_name}.*?(?=^class |\Z)'
    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(0).rstrip()
    return ""


def create_parameters_module(source_file: str, dest_file: str):
    """Extract Prm and ParameterBuilder to parameters.py."""
    with open(source_file, 'r') as f:
        content = f.read()

    # Extract Prm dataclass
    prm_class = extract_class(content, 'Prm')

    # Extract ParameterBuilder
    builder_class = extract_class(content, 'ParameterBuilder')

    module_content = '''"""Parameter handling and CLI parsing."""

import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Any, List
import numpy as np

from .config import Config
from .constants import GRIDSTEP, HELP_DESCRIPTION


'''

    module_content += prm_class + '\n\n\n' + builder_class

    with open(dest_file, 'w') as f:
        f.write(module_content)

    print(f"✓ Created {dest_file}")


def create_readers_module(source_file: str, dest_file: str):
    """Extract DataReader to readers.py."""
    with open(source_file, 'r') as f:
        content = f.read()

    reader_class = extract_class(content, 'DataReader')

    module_content = '''"""Data reading from files and pickle directories."""

import os
import re
import pickle
import numpy as np
from typing import Optional

from .config import Config
from .parameters import Prm, ParameterBuilder


'''

    module_content += reader_class

    with open(dest_file, 'w') as f:
        f.write(module_content)

    print(f"✓ Created {dest_file}")


def create_processors_module(source_file: str, dest_file: str):
    """Extract XBPMProcessor and BPMProcessor to processors.py."""
    with open(source_file, 'r') as f:
        content = f.read()

    xbpm_class = extract_class(content, 'XBPMProcessor')
    bpm_class = extract_class(content, 'BPMProcessor')

    module_content = '''"""XBPM and BPM data processors."""

import numpy as np
import matplotlib.pyplot as plt

from .parameters import Prm
from .visualizers import PositionVisualizer
from .exporters import Exporter
from .constants import STD_ROI_SIZE, FIGDPI


'''

    module_content += xbpm_class + '\n\n\n' + bpm_class

    with open(dest_file, 'w') as f:
        f.write(module_content)

    print(f"✓ Created {dest_file}")


def create_visualizers_module(source_file: str, dest_file: str):
    """Extract visualizer classes to visualizers.py."""
    with open(source_file, 'r') as f:
        content = f.read()

    blademap_class = extract_class(content, 'BladeMapVisualizer')
    position_class = extract_class(content, 'PositionVisualizer')

    module_content = '''"""Visualization classes for blade maps and positions."""

import numpy as np
import matplotlib.pyplot as plt

from .parameters import Prm
from .processors import XBPMProcessor
from .constants import FIGDPI


'''

    module_content += blademap_class + '\n\n\n' + position_class

    with open(dest_file, 'w') as f:
        f.write(module_content)

    print(f"✓ Created {dest_file}")


def create_exporters_module(source_file: str, dest_file: str):
    """Extract Exporter class to exporters.py."""
    with open(source_file, 'r') as f:
        content = f.read()

    exporter_class = extract_class(content, 'Exporter')

    module_content = '''"""Data export functionality."""

import numpy as np

from .parameters import Prm


'''

    module_content += exporter_class

    with open(dest_file, 'w') as f:
        f.write(module_content)

    print(f"✓ Created {dest_file}")


def main():
    """Main modularization workflow."""
    base_dir = Path(__file__).parent
    source = base_dir / 'xbpm_bumps.py'
    core_dir = base_dir / 'xbpm_bumps' / 'core'

    print("Starting modularization...")
    print(f"Source: {source}")
    print(f"Target: {core_dir}")
    print()

    # Create modules
    create_parameters_module(str(source), str(core_dir / 'parameters.py'))
    create_readers_module(str(source), str(core_dir / 'readers.py'))
    create_processors_module(str(source), str(core_dir / 'processors.py'))
    create_visualizers_module(str(source), str(core_dir / 'visualizers.py'))
    create_exporters_module(str(source), str(core_dir / 'exporters.py'))

    print()
    print("✅ Modularization complete!")
    print()
    print("Next steps:")
    print("1. Review generated modules for import errors")
    print("2. Test imports: python -c 'from xbpm_bumps.core import *'")
    print("3. Create new CLI entry point")
    print("4. Update xbpm_bumps.py to use package imports")


if __name__ == '__main__':
    main()
