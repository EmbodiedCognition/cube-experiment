'''Eye tracking calibration code.

This displays a 3x3 calibration grid and outputs the coordinates of the grid to
a file for loading in ViewPoint (as a 'Custom Set Points' file). It also
provides functions to adjust the grid on the fly (shifting the grid or changing
the distances between grid points) - after such an adjustment, the custom points
must be reloaded in ViewPoint.

In principle, it should work for grids other than 3x3. It was briefly tried, but
things didn't seem to work as well, and it was easier to use a 3x3 than figure
out what was going wrong.

It's also currently set up to track the left eye.

Code was adapted from EyeTrackerCalibrationNVIS obtained from Dmitry Kit in July
2012. Most modifications were in constructing the grid and adding some
flexibility to the adjustments.

V1.1 2014-03-21 Reindented, renamed to NvisCalibration (Leif Johnson)
V1.0 2013-10-21 Commented file released (Matthew Tong)
V0.2 2013-1-20 Replaced methods of grid construction and made fine tuning easier. (Matthew Tong)
V0.1 Internal copy (Dmitry Kit, adapted from C++)
'''

import math
import pickle
import viz


class NvisCalibration:
    '''Class that displays a calibration grid for the NVIS systems.

    Public methods:
    __init__ - Sets the internal variables to their initial conditions.
    getOutput - Returns a string of whether calibration is on.
    updateOffset - Shifts the grid using standard wasd keybindings.
    updateDelta - Changes the spacing between points using standard wasd keybindings.
    show - Turns on the grid, destroys any prior grid, and writes the current grid to file.
    writeoutSettings - Saves the current offset, dx, and dy values to lastSettings.txt.
    loadSettings - Loads previous offset, dx, and dy values from lastSettings.txt.
    writeCustomCalPoints - Writes the custom calibration points for ViewPoint.
    hide - Hides the grid and returns the viewpoint to its prior state.
    getToggle - Returns whether calibration is on.
    toggleCalib - Toggles calibration between showing and hiding.
    setProportion - Sets dx and dy manually.
    setOffset - Sets offset manually.
    '''

    def __init__(self, save_dir):
        '''Sets the internal variables to their initial conditions.'''
        self.previousMatrix = None
        self.save_dir = save_dir
        self.pos = None
        self.pos_screen_left = None
        self.pos_screen_right = None

        self.cross_tex = viz.addTexture('cross.png')
        self.circle_tex = viz.addTexture('circle.png')

        self.z = 0
        self.dx = 0
        self.dy = 0
        self.stimScale = 10
        self.offset = [0, 0]
        self.distance = 100
        self.grid = None
        self.state = viz.OFF

        self.centerVertAngle = math.radians(-7)
        self.centerHorizAngle = math.radians(-2)
        self.horizAngle = math.radians(17)
        self.vertAngle = math.radians(13)

        self.offset = [0, 0]
        self.numCols = 3
        self.numRows = 3
        self.adjustDelta = .5

    def getOutput(self):
        '''Returns a string of whether calibration is on.'''
        return 'C' if self.getToggle() else 'N'

    def updateOffset(self, key):
        '''Shifts the grid using standard wasd keybindings.'''
        if (self.state == viz.OFF): return
        if key == 'a':
            self.offset[0] -= self.adjustDelta
        elif key =='d':
            self.offset[0] += self.adjustDelta
        elif key == 'w':
            self.offset[1] += self.adjustDelta
        elif key =='s':
            self.offset[1] -= self.adjustDelta
        self.show()

    def updateDelta(self, key):
        '''Changes the spacing between points using standard wasd keybindings.'''
        if (self.state == viz.OFF): return
        if key == 'a':
            self.dx -= self.adjustDelta
        elif key =='d':
            self.dx += self.adjustDelta
        elif key == 'w':
            self.dy += self.adjustDelta
        elif key =='s':
            self.dy -= self.adjustDelta
        self.show()

    def show(self):
        '''Turns on the grid, destroys any prior grid, and writes the current grid to file.'''
        if not self.previousMatrix:
            self.previousMatrix = viz.MainView.getMatrix()

        self.state = viz.ON
        viz.MainView.reset(viz.HEAD_POS | viz.HEAD_ORI)

        if self.grid:
            self.grid.remove()
        self.grid = viz.addGroup()

        m =  viz.MainWindow.getMatrix()

        horizOffset = math.tan(self.centerHorizAngle) * self.distance + self.offset[0]
        vertOffset = math.tan(self.centerVertAngle) * self.distance + self.offset[1]

        horizDelta = math.tan(self.horizAngle) * self.distance + self.dx
        vertDelta = math.tan(self.vertAngle) * self.distance + self.dy

        topLeftPos = [horizOffset - horizDelta * (self.numCols -1) / 2,
                      vertOffset - vertDelta * (self.numRows - 1) / 2,
                      self.distance]

        self.pos = [[[0, 0, 0]] * self.numRows for k in range(self.numCols)]
        self.pos_screen_left = [[[0, 0, 0]] * self.numRows for k in range(self.numCols)]
        self.pos_screen_right = [[[0, 0, 0]] * self.numRows for k in range(self.numCols)]

        for i in range(0, self.numCols):
            for j in range(0, self.numRows):
                self.pos[i][j] = [topLeftPos[0] + i * horizDelta,
                                  topLeftPos[1] + j * vertDelta,
                                  self.distance]

                pt = viz.MainWindow.worldToScreen(self.pos[i][j], eye=viz.LEFT_EYE)
                pt[2] = 0
                self.pos_screen_left[i][j] = pt

                pt = viz.MainWindow.worldToScreen(self.pos[i][j], eye=viz.RIGHT_EYE)
                pt[2] = 0
                self.pos_screen_right[i][j] = pt

                q = viz.addTexQuad(parent=self.grid)
                q.texture(self.cross_tex)
                q.setScale(self.stimScale, self.stimScale, 1)
                q.setPosition(self.pos[i][j])
                q.billboard() # Makes always visible to viewer

        self.writeoutSettings()
        self.writeCustomCalPoints()

    def writeoutSettings(self):
        '''Saves the current offset, dx, and dy values  to lastSettings.txt.'''
        with open(self.save_dir +'/lastSettings.txt', 'w') as fp:
            pickle.dump([self.offset, self.dx, self.dy], fp)

    def loadSettings(self):
        '''Loads previous offset, dx, and dy values from lastSettings.txt.'''
        try:
            with open(self.save_dir + '/lastSettings.txt', 'r') as fp:
                self.offset, self.dx, self.dy = pickle.load(fp)
        except IOError as e:
            print ('Unable to open lastSettings.txt for use in EyeTrackerCalibration.'
                   "It's supposedly in " + self.save_dir)

    def writeCustomCalPoints(self):
        '''Writes the custom calibration points for ViewPoint.

        Created files are customCalPointsX.txt where X is Bi, R, or L for the
        different real and virtual eyes.
        '''
        for i, suffix in enumerate(('L', 'R', 'Bi')):
            fn = self.save_dir + '/customCalPoints%s.txt' % suffix
            fp = open(fn, 'w')
            if not fp:
                print 'fopen("%s", "w") failed'.format(fn)
                return

            fp.write('fkey_cmd 9 { clearHistory Both:calibration_PointDump }\n//\n')
            if i == 2:
                fp.write('binocular_Mode ON\n')
                fp.write('stereoDisplay ON\n//\n')
            else:
                fp.write('binocular_Mode OFF\n')
                fp.write('stereoDisplay OFF\n//\n')
            fp.write('calibration_Points %d\n' % (self.numCols * self.numRows))
            fp.write('calibration_PointLocationMethod Custom\n')
            fp.write('calibration_PresentationOrder Sequential\n')
            additional = int(i == 2)
            for k in range(0, 1 + additional):
                eye = 'EyeB'
                if i != 2:
                    eye = ''
                    if (i == 0):
                        e = self.pos_screen_left
                    else:
                        e = self.pos_screen_right

                elif k ==1:
                    eye = 'EyeB'
                    e = self.pos_screen_right
                else:
                    eye = 'EyeA'
                    e = self.pos_screen_left
                for m in range(0, self.numRows):
                    for l in range(0, self.numCols):
                        fp.write('%scalibration_CustomPoint  %d  %4.4f  %4.4f\n' %
                                 (eye, 1+(m*self.numCols+l),
                                  e[l][self.numRows - m - 1][0],
                                  1 - e[l][self.numRows - m - 1][1]))

            if i == 2:
                v1, v2, v5, v6 = 2*573, 1282+573, 'Secondary', '2'
            else:
                v1, v2, v5, v6 = 573, 1282, 'Custom', 'SceneCamera'

            if i==0:
                v3, v4 = 1920, 3200
            else:
                v3, v4 = 3200, 4480

            fp.write('//\n')
            fp.write('gazeSpaceGraphicsOptions -All\n')
            fp.write('gazeSpaceGraphicsOptions +Cal\n')
            fp.write('//\n')
            fp.write('calibration_WarningTime 5\n')
            fp.write('calibration_StimulusDuration 40\n')
            fp.write('calibration_ISI 2\n')
            fp.write('//\n')
            fp.write('stimulus_ImageHidden True\n')
            fp.write('stimulus_BackgroundColor    0    0    0\n')
            fp.write('calibration_BackgroundColor    0    0    0\n')
            fp.write('setWindow    CHILD    GazeSpace\n')
            fp.write('setWindow    SHOW    GazeSpace\n')
            fp.write('moveWindow    GazeSpace    700    1    %d    458\n' % v1)
            fp.write('moveWindow    Main    0    0    %d    1025\n' % v2)
            fp.write('stimWind_CustomStatic %d 0 %d 1024\n' % (v3, v4))
            fp.write('stimWind_FullDisplay %s\n' % v5)
            fp.write('viewSource %s\n' % v6)
            fp.write('END\n')
            fp.close()

    def hide(self):
        '''Hides the grid and returns the viewpoint to its prior state.'''
        viz.MainView.setMatrix(self.previousMatrix)
        self.previousMatrix = None
        self.state = viz.OFF

    def getToggle(self):
        '''Returns whether calibration is on.'''
        return self.state

    def toggleCalib(self):
        '''Toggles calibration between showing and hiding.'''
        if self.state == viz.OFF:
            self.show()
        else:
            self.hide()

    def setProportion(self,newDx, newDy):
        '''Sets dx and dy manually.'''
        self.dx2 = newDx
        self.dy2 = newDy
        self.show()

    def setOffset(self,newOffset):
        '''Sets offset manually.'''
        self.offset = newOffset
        self.show()
