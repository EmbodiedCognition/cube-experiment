'''VR lab utilities for Vizard.

This is designed to load in a system configuration file based on the name of the
computer it's being run on (as determined by platform.node()). If that file does
not exist, defaultSys.cfg is loaded.

It also sets up that system in a way that tries not to be tied to a particular
experiment to avoid each experiment replicating this work (potentially with
subtle differences).

Finally, it reads in an experiment config file, which must conform with the
configspec in expCfgSpec.ini.

V 1.2 Leif Johnson (refactored)

V 1.1 Matthew H. Tong - Made it so VRLabConfig could be created without an
  experiment config, making it easier to use purely to handle the system config.

V 1.0 Matthew H. Tong (adapted from other code from Dmitry Kit)
'''

import collections
import datetime
import os
import platform

import viz
import vizconfig

from . import phasespace
from . import hiball
from .configobj import ConfigObj
from .validate import Validator


class HMD:
    Stats = collections.namedtuple('Stats', 'offset overlap')

    def __init__(self, cfg, enabled=True):
        type = cfg['type']
        offset = cfg['offset']
        overlap = cfg['overlap']
        fov = cfg['fov']

        self.enabled = enabled
        self.hmd = None
        if not enabled:
            print 'HMD disabled'
            return

        if 'nvis' == type:
            import nvis
            self.hmd = nvis.nvisorSX111()
            print 'HMD type: NVIS SX111'

        elif 'oculus' == type:
            import oculus
            self.hmd = oculus.Rift()
            print 'HMD type: Oculus Rift'

        else:
            raise RuntimeError('Unsupported HMD type when starting HMD')

        vizconfig.register(self.hmd)

        if overlap < 0:
            overlap = 100

        self.stats = HMD.Stats(viz.MainWindow.getViewOffset, overlap)
        amount = self.stats.overlap - self.hmd._overlap
        self.hmd._overlap += amount
        offset = (self.hmd._hfov - self.hmd._overlap) / 2.0
        viz.MainWindow.setViewOffset(
            viz.Matrix.euler(-offset, 0.0, self.hmd._leftRollShift),
            eye=viz.LEFT_EYE)
        viz.MainWindow.setViewOffset(
            viz.Matrix.euler( offset, 0.0, self.hmd._rightRollShift),
            eye=viz.RIGHT_EYE)
        self.stats.offset = offset
        self.stats.overlap = self.hmd._overlap
        print 'Overlap', self.stats.overlap, 'Offset', self.stats.offset


class Config:
    '''This class represents the configuration for a particular experiment.

    It has two main components: vrlab.Config.sys and vrlab.Config.exp that
    contain the system configuration and the experiment configuration
    respectively.

    Attributes
    ----------
    sys : ConfigObj
        The system configuration
    exp : ConfigObj
        The experiment configuration

    writables : list
        A list of variables to be written to the .mov each frame.
    hmd : HMD
        Our HMD object.
    writer :
        The DVR writer. None if the writer is not being used or has not been
        initialized.
    eyetracker :
        The eyetracking interface. None if we aren't eyetracking.
    body_camera :
        A second virtual camera or tracker

    use_phasespace : bool
        True if phasespace is enabled.
    use_hmd: bool
        True if HMD is being used (otherwise configure just for screen).
    use_hiball: bool
        True if hiball is enabled.
    use_eyetracking : bool
        True if eyetracking is enabled.
    use_dvr : bool
        True if DVR should record the experiment.
    '''

    def __init__(self, filename=''):
        '''Opens and interprets configuration files for running experiments.

        This constructor opens both the system config (as defined by the
        <platform>.cfg file) and the experiment config (as defined by the given
        filename).

        Both configurations must conform the specs given in SYSTEM.ini and
        experiment.ini respectively. It also initializes the system as specified
        in the system config.
        '''
        self.exp = self.create_exp_config(filename)
        self.sys = self.create_sys_config()
        for path in self.sys['set_path']:
            viz.res.addPath(path)

        viz.window.setFullscreenMonitor(self.sys['displays'])
        viz.setMultiSample(self.sys['antiAliasPasses'])
        viz.MainWindow.clip(0.01, 200)

        self.writables = []
        self.writer = None
        self.mocap = None
        self.body_camera = None

        self._setup_hmd()
        self._setup_recorder()
        self._setup_eyetracking()
        self._setup_phasespace()
        self._setup_hiball()

        self.writables.append(self.mocap)
        self.writables.append(self.body_camera)

        if self.sys['use_fullscreen']:
            viz.go(viz.FULLSCREEN)
        else:
            viz.go()

    def create_sys_config(self):
        '''Set up the system config section.'''
        filename = '{}.cfg'.format(platform.node().upper())
        if not os.path.isfile(filename):
            filename = 'DEFAULT.cfg'
        print 'Loading system config file: {}'.format(filename)
        cfg = ConfigObj(filename, configspec='SYSTEM.ini', raise_errors=True)
        self.validate(cfg, 'System')
        print 'System config file parsed correctly'
        return cfg

    def create_exp_config(self, filename):
        '''Set up the experiment config section.'''
        if not filename:
            return None
        print 'Loading experiment config file: {}'.format(filename)
        cfg = ConfigObj(filename,
                        configspec='experiment.ini',
                        raise_errors=True,
                        file_error=True)
        self.validate(cfg, 'Experiment')
        print 'Experiment config file parsed correctly'
        for ld in cfg.get('_LOAD_', {}).get('loadList', ()):
            source = cfg['_LOAD_'][ld]
            print 'Loading: {} as {}'.format(ld, source['cfgFile'])
            c = ConfigObj(source['cfgFile'],
                          configspec=source['cfgSpec'],
                          raise_errors=True,
                          file_error=True)
            self.validate(c, 'Experiment[{}]'.format(ld))
            print 'Experiment config file parsed correctly'
            cfg.merge(c)
        return cfg

    def validate(self, cfg, which):
        if cfg.validate(Validator()):
            return
        errs = cfg.validate(Validator(), preserve_errors=True)
        for sections, key, err in configobj.flatten_errors(cfg, errs):
            sections.append(key or '[missing section]')
            print ', '.join(sections), ' = ', error or 'Missing value or section.'
        raise ValueError('{} config file validation failed!'.format(which))

    def _setup_hmd(self):
        if self.sys['use_hmd']:
            self.hmd = HMD(self.sys['hmd'], True)
            self.use_hmd = True
        else:
            self.hmd = HMD(self.sys['hmd'], False)
            self.use_hmd = False

    def _setup_recorder(self):
        if self.sys['use_DVR']:
            self.use_dvr = True
        else:
            self.use_dvr = False

    def _setup_eyetracking(self):
        if self.sys['use_eyetracking']:
            self.use_eyetracking = True
            if self.sys['hmd']['type'] != 'nvis':
                raise ValueError('Error in vrlab.Config: Eye-tracking not setup for this HMD.')
            import eyetracker
            self.eyetracker = eyetracker.NvisCalibration(self.sys['eyetracker']['settingsDir'])
            #self.eyetracker.distance = 10
            #self.eyetracker.fractionX = 4
            #self.eyetracker.fractionY = 4
            #self.eyetracker.stimScale = 5
            print 'Eye tracking enabled using NVIS HMD.'
        else:
            self.use_eyetracking = False
            self.eyetracker = None

    def _setup_phasespace(self):
        if self.sys['use_phasespace']:
            self.mocap = phasespace.Mocap(self.sys['phasespace']['server'])
            self.mocap.start_thread()
            self.use_phasespace = True
        else:
            self.use_phasespace = False

    def _setup_hiball(self):
        if self.sys['use_hiball']:
            cfg = self.sys['hiball']
            self.mocap = hiball.Camera(
                cfg['origin'],
                particle=None,
                sensorNum=cfg['headCam'],
                attachTo=viz.MainView,
                preTrans=cfg['preTransHead'])
            self.body_camera = None
            if cfg['bodyCam'] != -1:
                self.body_camera = hiball.Camera(
                    cfg['origin'],
                    particle=None,
                    sensorNum=cfg['bodyCam'],
                    attachTo=None,
                    preTrans=cfg['preTransBody'])
            self.use_hiball = True
        else:
            self.use_hiball = False

    def start(self):
        if not self.use_dvr:
            return

        print 'Starting DVR'
        from DVRwriter import DVRwriter

        # Can be filled in with useful metadata if desired.
        metadata = 'unused per-file meta data'

        # need to lazy initialize this because it has to be called after viz.go()
        if self.writer is None:
            root = os.path.join(self.sys['writer']['outFileDir'],
                                datetime.datetime.now().strftime('%Y.%m.%d.%H-%M'))

            self.exp.filename = '{}.exp.cfg'.format(root)
            self.sys.filename = '{}.sys.cfg'.format(root)

            width, height = viz.window.getSize()
            viewport = [width / 2, 0, width / 2, height]
            if 'L' == self.sys['eyetracker']['eye']:
                viewport[0] = 0

            print 'OutfileName:' + self.sys['writer']['outFileName']
            print 'Metadata:' + metadata
            print 'Viewport:' + str(viewport)
            print 'Eyetracking:' + str(self.use_eyetracking)

            self.writer = DVRwriter(
                '{}.{}' % (root, self.sys['writer']['outFileName']),
                metadata, viewport, self.use_eyetracking)

            self.exp.write()
            self.sys.write()

        self.writer.turnOn()

    def __record_data__(self, e):
        if self.use_dvr and self.writer != None:
            self.writer.write(self.writables)
