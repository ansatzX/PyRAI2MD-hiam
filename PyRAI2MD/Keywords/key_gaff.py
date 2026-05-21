######################################################
#
# PyRAI2MD 2 module for reading GAFF/GROMACS keywords
#
######################################################

import os
import sys

from PyRAI2MD.Utils.read_tools import ReadVal


class KeyGaff:

    def __init__(self):
        self.keywords = {
            'gmx': 'gmx',
            'mdp': '',
            'topol': '',
            'gro': '',
            'highlevel_topol': '',
            'highlevel_gro': '',
            'midlevel_topol': '',
            'midlevel_gro': '',
            'gaff_project': None,
            'gaff_workdir': os.getcwd(),
            'gaff_nproc': 1,
            'keep_tmp': 1,
            'verbose': 0,
        }

    def default(self):
        return self.keywords

    def update(self, values):
        keywords = self.keywords.copy()
        keyfunc = {
            'gmx': ReadVal('s'),
            'mdp': ReadVal('s'),
            'topol': ReadVal('s'),
            'gro': ReadVal('s'),
            'highlevel_topol': ReadVal('s'),
            'highlevel_gro': ReadVal('s'),
            'midlevel_topol': ReadVal('s'),
            'midlevel_gro': ReadVal('s'),
            'gaff_project': ReadVal('s'),
            'gaff_workdir': ReadVal('s'),
            'gaff_nproc': ReadVal('i'),
            'keep_tmp': ReadVal('i'),
            'verbose': ReadVal('i'),
        }

        for i in values:
            if len(i.split()) < 2:
                continue
            key, val = i.split()[0], i.split()[1:]
            key = key.lower()
            if key not in keyfunc:
                sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &gaff' % key)
            keywords[key] = keyfunc[key](val)

        return keywords

    @staticmethod
    def info(keywords):
        summary = """
  &gaff
-------------------------------------------------------
  GROMACS:                  %-10s
  GAFF_project:             %-10s
  GAFF_workdir:             %-10s
  MDP:                      %-10s
  Topology:                 %-10s
  GRO:                      %-10s
  Highlevel topology:       %-10s
  Highlevel GRO:            %-10s
  Midlevel topology:        %-10s
  Midlevel GRO:             %-10s
  Omp_num_threads:          %-10s
  Keep tmp_gaff:            %-10s
-------------------------------------------------------
    """ % (
            keywords['gmx'],
            keywords['gaff_project'],
            keywords['gaff_workdir'],
            keywords['mdp'],
            keywords['topol'],
            keywords['gro'],
            keywords['highlevel_topol'],
            keywords['highlevel_gro'],
            keywords['midlevel_topol'],
            keywords['midlevel_gro'],
            keywords['gaff_nproc'],
            keywords['keep_tmp'],
        )

        return summary
