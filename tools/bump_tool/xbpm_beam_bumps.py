#!/bin/env python3

import argparse
import math
import os
import sys
import time
import traceback
from datetime import datetime

import epics
import numpy as np

from apsuite.commisslib.bumps_scans import Bump, BumpParams
from mathphys.functions import save
from matplotlib import pyplot as plt
from siriuspy.devices import APU, CurrInfoSI, SOFB, VPU

EPICS_CA_ADDR_LIST = [
    "10.39.50.33",
    "10.39.50.34",
    "10.0.38.59:62000",
    "10.30.13.22:62000",
    "10.30.14.19:62000"
    ]
os.environ["EPICS_CA_ADDR_LIST"] = " ".join(EPICS_CA_ADDR_LIST)

BEAMLINES = {
    "CNB": {
        "prefix": "CNB:FE:PICO01",
        "device_type": "pico",
        "xbpm_numbers": (1,),
        "subsec": "06SB",
    },
    "CAT": {
        "prefix": "CAT:FE:PICO01",
        "device_type": "pico",
        "xbpm_numbers": (1,),
        "subsec": "07SP",
    },
    "MNC": {
        "prefix": "SI-09SAFE:DI-PBPM-",
        "device_type": "fmc",
        "xbpm_numbers": (1, 2),
        "subsec": "09SA",
    },
    "MGN": {
        "prefix": "SI-10BCFE:DI-PBPM-",
        "device_type": "fmc",
        "xbpm_numbers": (1, 2),
        "subsec": "10BC",
    },
}

bl_params = {
    'name': '',
    'gap_phase': 0,
}

acq_data = {
    'n_points': 11,
    'min_x': -10,
    'max_x':  10,
    'min_y': -10,
    'max_y':  10,
}

def cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="read XBPM beam bumps parameter file"
        )

    parser.add_argument(
        '-p', '--parameter_file', type=str, required=True,
        help='Parameter file name'
        )

    args = parser.parse_args()
    return args

def parameter_read(file_name: str):
    parsed_beamline = {'name': '', 'gap_phase': bl_params['gap_phase']}
    parsed_acq_data = acq_data.copy()
    converters = {
        'name': ('beamline', str),
        'gap_phase': ('beamline', float),
        'n_points': ('acq_data', int),
        'min_x': ('acq_data', float),
        'max_x': ('acq_data', float),
        'min_y': ('acq_data', float),
        'max_y': ('acq_data', float),
    }
    seen = set()

    with open(file_name, 'r') as pf:
        for line_number, line in enumerate(pf, start=1):
            line = line.split('#', maxsplit=1)[0].strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{file_name}:{line_number}: expected a key and a value"
                )
            key, value = parts
            if key not in converters:
                raise ValueError(f"{file_name}:{line_number}: unknown key {key!r}")
            if key in seen:
                raise ValueError(f"{file_name}:{line_number}: duplicate key {key!r}")
            seen.add(key)

            section, converter = converters[key]
            try:
                converted = converter(value)
            except ValueError as exc:
                raise ValueError(
                    f"{file_name}:{line_number}: invalid value for {key!r}: {value!r}"
                ) from exc
            if section == 'beamline':
                parsed_beamline[key] = converted
            else:
                parsed_acq_data[key] = converted

    if not parsed_beamline['name']:
        raise ValueError(f"{file_name}: missing required 'name' setting")
    if parsed_beamline['name'] not in BEAMLINES:
        raise ValueError(
            f"{file_name}: unknown beamline {parsed_beamline['name']!r}; "
            f"expected one of {', '.join(BEAMLINES)}"
        )
    if parsed_acq_data['n_points'] < 2:
        raise ValueError(f"{file_name}: 'n_points' must be at least 2")
    if parsed_acq_data['min_x'] >= parsed_acq_data['max_x']:
        raise ValueError(f"{file_name}: 'min_x' must be less than 'max_x'")
    if parsed_acq_data['min_y'] >= parsed_acq_data['max_y']:
        raise ValueError(f"{file_name}: 'min_y' must be less than 'max_y'")
    if not math.isfinite(parsed_beamline['gap_phase']):
        raise ValueError(f"{file_name}: 'gap_phase' must be finite")
    for key in ('min_x', 'max_x', 'min_y', 'max_y'):
        if not math.isfinite(parsed_acq_data[key]):
            raise ValueError(f"{file_name}: {key!r} must be finite")

    bl_params.update(parsed_beamline)
    acq_data.update(parsed_acq_data)


class XBPM:
    def __new__(cls, name: str, xbpm_nr: int):
        if cls is XBPM:
            config = BEAMLINES.get(name)
            if config is None:
                raise ValueError(f"Unknown beamline name: {name}")
            device_class = PICO if config["device_type"] == "pico" else FMC
            return super().__new__(device_class)
        return super().__new__(cls)

    def __init__(self, name: str, xbpm_nr: int):
        if xbpm_nr not in BEAMLINES[name]["xbpm_numbers"]:
            raise ValueError(
                f"XBPM number {xbpm_nr} is not configured for beamline {name}"
            )
        self.prefix = BEAMLINES[name]["prefix"]
        self.acq_rate = 10 # Hz

    def photon_shutter(self, channel_number: int):
        return None

    def range_value(self, channel_number: int):
        return None

    def saturation(self, channel_number: int):
        return None


class FMC(XBPM):
    def __init__(self, name: str, xbpm_nr: int):
        super().__init__(name, xbpm_nr)
        self.prefix = self.prefix + str(xbpm_nr)

    def ampA(self):
        return epics.caget(self.prefix + ':AmplA-Mon')

    def ampB(self):
        return epics.caget(self.prefix + ':AmplB-Mon')

    def ampC(self):
        return epics.caget(self.prefix + ':AmplC-Mon')

    def ampD(self):
        return epics.caget(self.prefix + ':AmplD-Mon')
    
    def posX(self):
        return epics.caget(self.prefix + ':PosX-Mon')

    def posY(self):
        return epics.caget(self.prefix + ':PosY-Mon')


class PICO(XBPM):
    def __init__(self, name: str, xbpm_nr: int):
        super().__init__(name, xbpm_nr)

    def ampA(self):
        return epics.caget(self.prefix + ':Current1:EngValue'), epics.caget(
            self.prefix + ':Current1:EngUnit'
        )

    def ampB(self):
        return epics.caget(self.prefix + ':Current2:EngValue'), epics.caget(
            self.prefix + ':Current2:EngUnit'
        )

    def ampC(self):
        return epics.caget(self.prefix + ':Current3:EngValue'), epics.caget(
            self.prefix + ':Current3:EngUnit'
        )

    def ampD(self):
        return epics.caget(self.prefix + ':Current4:EngValue'), epics.caget(
            self.prefix + ':Current4:EngUnit'
        )

    def range_value(self, channel_number: int):
        return epics.caget(
            f'{self.prefix}:Current{channel_number}:Range_RBV'
        )

    def saturation(self, channel_number: int):
        return epics.caget(
            f'{self.prefix}:Current{channel_number}:Saturated'
        )


    def check_acq_freeze(self, period: int, threshold: int) -> bool:
        """Return True if all channels are identical for ``threshold`` samples."""
        if period < 1:
            raise ValueError("period must be at least one second")
        if threshold < 1 or threshold > period:
            raise ValueError("threshold must be between 1 and period")

        readers = (self.ampA, self.ampB, self.ampC, self.ampD)

        def read_data():
            data = tuple(read() for read in readers)
            if any(value is None for channel in data for value in channel):
                raise RuntimeError("Unable to read all XBPM acquisition PVs")
            return data

        previous_data = read_data()
        identical_count = 0
        for _ in range(period):
            time.sleep(1)
            data = read_data()
            if data == previous_data:
                identical_count += 1
                if identical_count == threshold:
                    print(
                        "This XBPM electronics could be frozen or turned off. "
                        "Please check the device."
                    )
                    return True
            else:
                identical_count = 0
            previous_data = data

        print("No freeze was detected during the observation period.")
        return False


def set_id_position(name, gap_phase):
    if name == 'MNC':
        apu_mnc.set_gap_phase(gap_phase)
    elif name == 'CNB':
        vpu_cnb.set_gap_phase(gap_phase)
    elif name == 'CAT':
        vpu_cat.set_gap_phase(gap_phase)
    elif name == 'MGN':
        print("MGN is a BC beamline. No further configuration is required.")
    else:
        raise ValueError(f"Unknown beamline name: {name}")


def configure_grid(show=False):
    angsx = np.linspace(acq_data['min_x'], acq_data['max_x'], acq_data['n_points'])
    angsy = np.linspace(acq_data['min_y'], acq_data['max_y'], acq_data['n_points'])

    x, y = np.meshgrid(angsx, angsy)

    if show:
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, 'o')
        plt.title('Scan grid')
        plt.xlabel('X [µm]')
        plt.ylabel('Y [µm]')
        plt.show()

    return x, y

# To be finished
def check_interlocks(pvs: list) -> bool:
    for pv in pvs:
        value = epics.caget(pv)
        if value is None:
            print(f"Warning: PV {pv} is not responding.")
            return False
        if value != 0:
            print(f"Interlock triggered on PV {pv}. Value: {value}")
            return False
    return True


def get_meas_data(
    blades: dict,
    positions: dict,
    currinfo: CurrInfoSI,
    sofb: SOFB,
    idlist: list,
    subsec:str,
    agx=0,
    agy=0,
    psx=0,
    psy=0,
    ):  
    
    idinfo = {
        'mnc': idlist[0].phase_mon,
        'cnb': idlist[1].gap_mon,
        'cat': idlist[2].gap_mon,
    }

    machinfo = {
        'current': currinfo.current,
        'agx': agx,
        'agy': agy,
        'posx': psx,
        'posy': psy,
        'orbx': sofb.orbx,
        'orby': sofb.orby,
    }

    data = (blades, idinfo, machinfo)

    acq_time = time.time()
    fname = 'xbpm_acq_subsec:'
    fname += f'{subsec}_agx:{agx:03.0f}_agy:{agy:03.0f}_'
    fname += 'time_' + str(acq_time) + '.pickle'

    p_fname = 'xbpm_posmeas_subsec:'
    p_fname += f'{subsec}_agx:{agx:03.0f}_agy:{agy:03.0f}_'
    p_fname += 'time_' + str(acq_time) + '.pickle'

    save(data, fname)
    save(positions, p_fname)

    return data


def get_xbpm_readings(
        xbpmlist: list, 
        names: list,
        subsec: str,
        nr_meas=10, 
        tinterval=1,
        agx=0,
        agy=0,
        psx=0,
        psy=0,
    ):
    blades    = {}
    positions = {}
    
    for xbpm, name in zip(xbpmlist, names):
        blades[name] = {
            'prefix': xbpm.prefix,
            'A_val': [],
            'B_val': [],
            'C_val': [],
            'D_val': [],
            'A_range': [],
            'B_range': [],
            'C_range': [],
            'D_range': [],
            'A_saturation': [],
            'B_saturation': [],
            'C_saturation': [],
            'D_saturation': [],
        }
        positions[name] = {
                'prefix': xbpm.prefix,
                'X_pos' : [],
                'Y_pos' : [],
            }

    for _ in range(nr_meas):
        for xbpm, name in zip(xbpmlist, names):
            for channel_number, channel in enumerate('ABCD', start=1):
                blades[name][f'{channel}_val'].append(
                    getattr(xbpm, f'amp{channel}')()
                )
                blades[name][f'{channel}_range'].append(
                    xbpm.range_value(channel_number)
                )
                blades[name][f'{channel}_saturation'].append(
                    xbpm.saturation(channel_number)
                )

            positions[name]['X_pos'].append(xbpm.posX())
            positions[name]['Y_pos'].append(xbpm.posY())

        time.sleep(tinterval)
    print(f'{subsec}_agx:{agx:03.0f}_agy:{agy:03.0f}')
    get_meas_data(blades, positions, currinfo, sofb, idlist, subsec, agx, agy, psx, psy)

    return blades


def run_bump_scan(bump: Bump, beamline_name: str) -> None:
    bump.get_initial_state()
    try:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M")
        ref_orb_bumps = np.concatenate((bump.reforbx, bump.reforby))

        print('Saving reference orbit to file...')
        save(ref_orb_bumps, f'ref_orb_bumps_{beamline_name}_{current_time}.pickle')

        steps = (
            ("Continue to SOFB configuration? (Y/N)", bump.config_sofb),
            ("Remove BPMs from SOFB? (Y/N)", bump.remove_bpms),
            ("Start scan? (Y/N)", bump.do_scan),
        )
        for prompt, action in steps:
            if input(prompt).strip().upper() != "Y":
                print('Aborting operation.')
                return
            action()
        print('Acquisition finished.')
    finally:
        print('Restoring initial state...')
        bump.restore_initial_state()
        print('Done!')


if __name__ == '__main__':
    try:
        sofb = SOFB(SOFB.DEVICES.SI)
        currinfo = CurrInfoSI()

        apu_mnc = APU(APU.DEVICES.APU22_09SA)
        vpu_cnb = VPU(VPU.DEVICES.VPU29_06SB)
        vpu_cat = VPU(VPU.DEVICES.VPU29_07SP)
        idlist = [apu_mnc, vpu_cnb, vpu_cat]

        args = cmd_args()
        parameter_read(args.parameter_file)

        set_id_position(bl_params['name'], bl_params['gap_phase'])
        beamline_config = BEAMLINES[bl_params['name']]
        xbpm_numbers = beamline_config['xbpm_numbers']
        xbpms = [XBPM(bl_params['name'], number) for number in xbpm_numbers]
        xbpm_names = [f"{bl_params['name']}{number}" for number in xbpm_numbers]
        angsx, angsy = configure_grid(show=True)

        check_interlocks()
        if isinstance(xbpms[0], PICO):
            xbpms[0].check_acq_freeze(period=120, threshold=10)

        bump_param = BumpParams()
        bump_param.subsec = beamline_config['subsec']
        bump_param.pts_agx = angsx
        bump_param.pts_agy = angsy
        bump_param.do_angular_bumps = True
        bump_param.closed_loops = True
        bump_param.n_bpms_outx = 2
        bump_param.n_bpms_outy = 2
        bump_param.minsingval = 1
        bump_param.bump_residue = 1
        bump_param.bump_max_residue = 10
        bump_param.wait_meas = 0.5
        bump_param.meas_func = get_xbpm_readings
        bump_param.kwargs = {
            'nr_meas': 10,
            'tinterval': 1,
            'subsec': beamline_config['subsec'],
        }

        bump_param.args = [xbpms, xbpm_names]

        bump = Bump(bump_param)
        run_bump_scan(bump, bl_params['name'])
    except SystemExit as exc:
        if exc.code not in (0, None):
            sys.exit(1)
        raise
    except Exception as exc:
        print(f"Erro durante a execução do script: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
