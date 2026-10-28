#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 || ! $1 =~ ^[A-Z]{3}$ ]]; then
    echo "Usage: $0 <beamline name> <gap/phase>" >&2
    exit 2
fi

beamline=$1
shift
gap_phase=($@)

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config_file=/tmp/xbpm_bump_parameters.txt

create_config_file() {
    local gap_param="$1"
    local config_file="/tmp/xbpm_bump_parameters_${gap_param}.txt"
    echo "Creating configuration file: $config_file"
    cat > "$config_file" <<EOF
# Beamline Parameters
# Prefix, XBPM devices, and subsec are derived from the beamline name.
name $beamline
gap_phase $gap_param

# Acquisition Grid Parameters
n_points 15
min_x -20
max_x 20
min_y -20
max_y 20
EOF
}

handle_gaps_list() {
    local arr_size=${#gap_phase[@]}
    local counter=0

    if [[ ${#gap_phase[@]} -gt 1 ]]; then
        for gap in "${gap_phase[@]}"; do
            ((++counter))

            if (( counter == arr_size )); then
                echo "Running beam bumps for gap/phase: $gap"
                create_config_file "$gap"
                python3 "$script_dir/xbpm_beam_bumps.py" --parameter_file "$config_file"
            else
                echo "Running beam bumps for gap/phase: $gap"
                create_config_file "$gap"
                python3 "$script_dir/xbpm_beam_bumps.py" --parameter_file "$config_file"
                echo "Moving to next gap/phase..."
            fi

        done
        echo "All specified gap/phase values have been processed."
    else
        echo "Only one gap/phase value provided. Processing: ${gap_phase[0]}"
        create_config_file "${gap_phase[0]}"
        python3 "$script_dir/xbpm_beam_bumps.py" --parameter_file "$config_file"
    fi
}

handle_gaps_list
